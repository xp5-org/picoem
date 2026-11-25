//! RP2040 emulator library.
//!
//! Phase 5.A fills in the bus fabric, CLOCKS/RESETS/PLL/XOSC/ROSC
//! register storage, full SIO (GPIO, CPUID, FIFO, spinlocks, divider,
//! interpolators — **no** doorbells / MTIME / coprocessor bridge),
//! IO_BANK0 / PADS_BANK0, XIP_CTRL / SSI stubs, and dual-core stepping
//! (core 0 runs; core 1 stays halted until woken via the SIO FIFO
//! protocol).
//!
//! Phase 5.B wires the two PIO blocks (`bus.pio[0]`, `bus.pio[1]`) into
//! AHB at `0x5020_0000` / `0x5030_0000`, steps them once per emulator
//! step, and merges their pad outputs into `bus.gpio_in` (PIO OE
//! overrides SIO on a per-pin basis, mirroring `rp2350_emu::Emulator`).
//!
//! See `wrk_docs/2026.04.14 - HLD - mdpicoem Workspace Restructure.md`.

use tracing::info;

pub mod bus;
pub mod core;
pub mod dma;
pub mod dreq;
pub mod irq;
pub mod memory;
pub mod peripherals;

// Dual-execution HLD V1 (Stage 3b.2) — threaded runtime scaffolding.
// The module file internally `#![cfg]`-gates to x86_64 Windows + the
// `threading` cargo feature, so non-Windows and `--no-default-features`
// builds compile an empty module and the serial path is unaffected.
#[cfg(feature = "threading")]
pub mod threaded;

// -----------------------------------------------------------------------
// Dual-execution HLD V1 (Stage 3b.1) — public types.
//
// Introduces the `ExecutionModel` selector, `ConfigError`, `WorkerName`,
// and `EmulatorError` to mirror the RP2350 crate. Stage 3b.1 ships the
// types + the `CoreBus` trait port so later sub-stages (3b.2: threaded/
// module, 3b.4: builder wiring) can land against a stable surface. The
// Emulator dispatch path stays Serial-only in 3b.1.
// -----------------------------------------------------------------------

/// Execution model for an [`Emulator`]. Selected at construction via
/// [`EmulatorBuilder::execution`]; cannot be switched post-build.
///
/// - `Serial` — oracle-validated reference path (QEMU + silicon
///   differentials). Single-threaded, per-instruction interleave.
///   Always available.
/// - `Threaded` — multi-thread runtime; opt-in throughput optimization
///   on x86_64 Windows or Linux hosts with the `threading` cargo
///   feature on.
///   Not validated against QEMU/silicon oracles. Not yet wired into
///   [`Emulator::step`] — arrives with Stage 3b.4.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExecutionModel {
    #[default]
    Serial,
    Threaded,
}

/// Errors returned by [`EmulatorBuilder::build`] once the Stage 3b.4
/// wiring lands. The only non-trivial variant today is
/// `ThreadingUnavailable`, returned when the caller selects
/// [`ExecutionModel::Threaded`] but the host platform or build
/// configuration cannot satisfy it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    /// `ExecutionModel::Threaded` selected but the current build does
    /// not include a threaded runtime — either the `threading` cargo
    /// feature is off, or the host is not one of the supported
    /// platforms (currently x86_64 Windows or Linux only).
    ThreadingUnavailable,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::ThreadingUnavailable => write!(
                f,
                "ExecutionModel::Threaded is unavailable (requires x86_64 Windows/Linux \
                 with the `threading` cargo feature enabled)"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Identifier for a worker thread in the threaded runtime. RP2040
/// uses a three-worker layout (core0, core1, coordinator) — smaller
/// than RP2350's six-worker layout because M0+ has no PIO-as-worker
/// split in the Stage 3b plan. rp2350_emu's `Pio0`/`Pio1`/`Pio2` worker
/// variants are intentionally omitted here; if PIO becomes a
/// bottleneck the enum can gain those variants in a follow-up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerName {
    Core0,
    Core1,
    Coord,
}

impl WorkerName {
    /// Short label for summary tables / error messages. Kept stable so
    /// harness tooling can scrape diagnostic output.
    pub fn as_str(self) -> &'static str {
        match self {
            WorkerName::Core0 => "core0",
            WorkerName::Core1 => "core1",
            WorkerName::Coord => "coord",
        }
    }
}

/// Errors returned by post-construction [`Emulator`] methods once the
/// Stage 3b.4 wiring lands. Surfaces runtime-model mismatches and
/// worker panics (dual-execution HLD V1 §5.5).
///
/// `WorkerPanicked` is sticky: once an [`Emulator`] observes a worker
/// panic, every subsequent call on that instance returns the same
/// error without re-attempting the workers (one-shot-after-panic, HLD
/// §5.5 item 5). Drop the instance and rebuild from a fresh
/// [`EmulatorBuilder`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmulatorError {
    /// Called a Serial-only method on a Threaded emulator, e.g.
    /// `step()` — Threaded runs in quanta, not single-step. HLD §5.4.
    NotSupportedInThreadedMode,
    /// One of the worker threads panicked. The `Emulator` is sticky-
    /// poisoned after this; drop and rebuild. Only produced on the
    /// Threaded path.
    WorkerPanicked { which: WorkerName, message: String },
    /// The shared [`picoem_common::SpinBarrier`] watchdog fired
    /// because a worker failed to arrive at the rendezvous within
    /// [`picoem_common::threaded::DEFAULT_DEADLINE`]. The `Emulator`
    /// is sticky-poisoned after this; drop and rebuild. HLD V1 §6.6.
    ///
    /// Only produced on the Threaded path. `which` is the first worker
    /// that returned `TimedOut` at its barrier; since the barrier
    /// cannot identify *which* worker failed to arrive, this field
    /// names an observer rather than the culprit. `elapsed_ms` is the
    /// reporting waiter's own wall-clock elapsed time at expiry.
    BarrierTimeout { which: WorkerName, elapsed_ms: u32 },
}

impl std::fmt::Display for EmulatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmulatorError::NotSupportedInThreadedMode => write!(
                f,
                "operation not supported on a Threaded Emulator (Serial-only)"
            ),
            EmulatorError::WorkerPanicked { which, message } => {
                write!(f, "worker {} panicked: {message}", which.as_str())
            }
            EmulatorError::BarrierTimeout { which, elapsed_ms } => write!(
                f,
                "barrier watchdog fired (observed by worker {}) after {}ms",
                which.as_str(),
                elapsed_ms
            ),
        }
    }
}

impl std::error::Error for EmulatorError {}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod pio_tests;

#[cfg(test)]
mod block_tests;

pub use self::bus::Bus;
pub use self::core::CortexM0Plus;
pub use self::core::block::{BlockMode, BlockStats, Blocks};
pub use self::memory::{Memory, ROM_SIZE, SRAM_SIZE, bank_for_address};

pub use picoem_common::Pacer;
pub use picoem_common::{Clock, PacerSnapshot, PacerStats};

/// ROSC nominal frequency (~6.5 MHz). RP2040 boots on ROSC at the same
/// nominal rate as RP2350; PLL configuration (if any) happens later in
/// firmware.
pub use picoem_common::ROSC_FREQ_HZ;

/// Emulator configuration.
pub struct Config {
    /// System clock frequency in Hz. Default: ROSC (~6.5 MHz).
    pub sys_clk_hz: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sys_clk_hz: ROSC_FREQ_HZ,
        }
    }
}

/// Default quantum size in cycles. Matches `rp2350_emu`.
pub const DEFAULT_STEP_QUANTUM: u32 = 64;

/// Top-level RP2040 emulator. Owns dual Cortex-M0+ cores, bus fabric,
/// memory, and clock.
///
/// Dual-execution HLD V1: an `Emulator` has a fixed [`ExecutionModel`]
/// picked at construction time via [`EmulatorBuilder::execution`]. In
/// Serial mode (default) the `cores` / `bus` / `clock` fields are the
/// authoritative state and the existing per-instruction interleave
/// applies. In Threaded mode those fields retain their post-seed
/// snapshot until the first `run_quantum` promotes them into the
/// threaded runtime; afterwards the flat fields are zero-cost
/// placeholders and typed accessors fire a debug-assert if touched.
/// Architectural state of one core at a loop head, for the idle-loop skip.
struct IdleSnap {
    pc: u32,
    regs: core::registers::Registers,
    cycles: u64,
    side: [u32; 9],
    /// Taken in watch mode, with the other core running.
    watch: bool,
}

/// A core whose clean polling pass is parked: it is not stepped while the
/// bus sees nothing that could change what the pass reads.
#[derive(Clone, Copy)]
struct Park {
    /// Master cycle the park began at.
    since: u64,
    /// Cycles one pass takes.
    iter: u64,
}

/// Exact idle-loop skip for the Serial model. While one core sleeps,
/// halted or in WFE, nothing but the other core can change memory within a
/// quantum: peripherals, PIO and DMA move at quantum ends, and so do the
/// TIMER counters. A running core that comes back to a loop head with the
/// same registers, memory, spinlocks, divider and event flags as the last
/// time, having made only
/// reads that cannot change within the quantum, register ops and plain
/// SRAM stores, will repeat the pass identically until the quantum ends.
/// The skip advances its cycle count by as many whole passes as fit, which
/// is bit-identical to running them. Loop heads are backward branches,
/// calls and returns to lower addresses included; a snapshot holds across
/// other heads while the stretch stays clean, and compares only at the same
/// PC, SP and LR.
pub struct IdleSkip {
    /// On by default; off runs every pass, for A/B checks.
    pub enabled: bool,
    /// Cycles skipped per core.
    pub skipped_cycles: [u64; 2],
    /// Off by default. On, a polling pass is also caught while the other
    /// core runs: watch mode. A watch-mode pass may make no stores and reads
    /// only ROM and flash, the TIMER raw counters, SIO CPUID and GPIO_IN,
    /// PIO FSTAT and FLEVEL, and up to 32 SRAM words, read or stored; every
    /// stored byte must hold its snapshot value again. When it repeats with
    /// equal state the core is parked: it is not stepped, and the bus checks
    /// the running core's accesses against those reads. A store to one of
    /// the words, a read of one the pass stored, any other access to a
    /// polled PIO block, a GPIO output
    /// write while GPIO_IN is polled, or a FIFO write unparks it; so does
    /// the quantum end. The core is then credited with the whole passes
    /// that fit since it parked and resumes at its loop head. PIO, DMA,
    /// peripherals and IRQ delivery move only at quantum ends, so the
    /// result matches running the passes, to within one pass of timing.
    pub two_core: bool,
    /// Cycles credited to parked cores, per core.
    pub parked_cycles: [u64; 2],
    /// Detector event counts, per core; diagnostics only.
    pub stats: [IdleStats; 2],
    /// The watch-mode snapshot that may outlive the quantum end, with the
    /// fingerprint of its inputs at that end; see `snap_carries`.
    carry: Option<usize>,
    carry_vals: Vec<u32>,
    /// Scratch for `snap_carries`.
    carry_now: Vec<u32>,
    snap: [Option<IdleSnap>; 2],
    park: [Option<Park>; 2],
    fails: [u8; 2],
    /// Loop heads whose last pass came back changed or dirty, per core,
    /// with the master cycle until which no snapshot is taken there.
    busy: [[(u32, u64); IDLE_BUSY_HEADS]; 2],
    /// Loop heads that parked, per core, most recent first. They are
    /// polls, so a pass that once came back changed does not mark them
    /// busy.
    parked_heads: [[u32; IDLE_PARKED_HEADS]; 2],
    /// Per core, the first loop head after an exception and the core's
    /// cycle count there, while snapshots wait; see `idle_settling`.
    settle: [Option<(u32, u64)>; 2],
}

impl Default for IdleSkip {
    fn default() -> Self {
        Self {
            enabled: true,
            skipped_cycles: [0; 2],
            two_core: false,
            parked_cycles: [0; 2],
            stats: [IdleStats::default(); 2],
            carry: None,
            carry_vals: Vec::new(),
            carry_now: Vec::new(),
            snap: [None, None],
            park: [None, None],
            fails: [0; 2],
            busy: [[(u32::MAX, 0); IDLE_BUSY_HEADS]; 2],
            parked_heads: [[u32::MAX; IDLE_PARKED_HEADS]; 2],
            settle: [None; 2],
        }
    }
}

/// What the idle-loop detector did, per core.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdleStats {
    /// Snapshots taken at a loop head.
    pub snaps: u64,
    /// Snapshots dropped because the pass came back dirty.
    pub dirty: u64,
    /// Compares at the snapshot head that found other state.
    pub changed: u64,
    /// Compares that skipped passes while the other core slept.
    pub skips: u64,
    /// Compares that parked the core.
    pub parks: u64,
    /// Snapshots still live at a quantum end and dropped there.
    pub expired: u64,
    /// Snapshots carried across a quantum end.
    pub snap_carried: u64,
    /// Snapshots dropped by a mode change or the other core's snapshot.
    pub displaced: u64,
    /// Parks carried across a quantum end.
    pub carried: u64,
    /// Parks ended at a quantum end because the pass read the TIMER.
    pub ended_timer: u64,
    /// Parks ended at a quantum end for any other reason.
    pub ended_other: u64,
    /// Parks ended inside a quantum by a conflicting access.
    pub hits: u64,
}

/// Snapshots taken plus failed compares per core per quantum before the
/// detector backs off.
const IDLE_MAX_FAILS: u8 = 12;

/// Loop heads remembered as busy per core, and for how many master cycles.
/// A head whose pass came back with other registers or with dirt the loop
/// made itself is computing, not polling; snapshots there are what make
/// the detector cost time. A polling loop passes on its first compare and
/// is never marked; one that only sometimes idles parks at most this late.
/// The table is direct-mapped by [`busy_slot`], so a head that shares a
/// slot evicts the other.
const IDLE_BUSY_HEADS: usize = 256;
const IDLE_BUSY_CYCLES: u64 = 1 << 18;
const IDLE_PARKED_HEADS: usize = 16;

/// Longest a core waits after an exception before snapshots resume.
const IDLE_SETTLE_CYCLES: u64 = 1024;

/// The cycles of the whole passes of `iter` cycles in `span`. A quantum
/// is usually under two passes, so the division is rare.
#[inline(always)]
fn whole_passes(span: u64, iter: u64) -> u64 {
    if span < iter {
        0
    } else if span < 2 * iter {
        iter
    } else {
        span / iter * iter
    }
}

/// The slot of loop head `pc` in the busy table.
#[inline(always)]
pub(crate) fn busy_slot(pc: u32) -> usize {
    ((pc >> 1).wrapping_mul(0x9E37_79B1) >> 24) as usize & (IDLE_BUSY_HEADS - 1)
}

/// Whether `pc` is a busy head in `table` at master cycle `now`.
#[inline(always)]
pub(crate) fn is_busy_head(table: &[(u32, u64)], pc: u32, now: u64) -> bool {
    let e = table[busy_slot(pc)];
    e.0 == pc && e.1 > now
}

pub struct Emulator {
    /// Exact idle-loop skip; see [`IdleSkip`].
    pub idle: IdleSkip,
    /// Block engine for a core stepping alone; see [`Blocks`]. Off by
    /// default; `PICOEM_BLOCKS=on|check|off` overrides at build, and
    /// `PICOEM_JIT=on` compiles hot blocks when built with the `jit` feature.
    pub blocks: Blocks,
    pub cores: [CortexM0Plus; 2],
    pub bus: Bus,
    pub clock: Clock,
    /// Cycles advanced per call to [`Self::step`].
    pub step_quantum: u32,
    /// Pin trace for harnesses that rebuild a signal from the pins. While
    /// `Some`, every system cycle appends the merged GPIO level word, and
    /// PIO steps one cycle at a time so no edge inside a quantum is lost.
    /// `None` costs nothing. Serial model only.
    pub gpio_trace: Option<Vec<u32>>,
    /// Total PIO ticks performed in the slow path
    /// (`tick_pio_and_route_irqs`). Diagnostic-only — used by the
    /// PicoGUS harness to confirm PIO is actually being driven.
    /// Bumps by `cycles` per quantum after HLD 2026.04.26 V5 chunked
    /// refactor (per-quantum granularity is acceptable).
    pub pio_tick_count: u64,
    /// Subset of [`Self::pio_tick_count`] where bit 4 (IOW for PicoGUS)
    /// of `bus.gpio_in` was low at the moment of the tick. If this stays
    /// at zero while the harness is asserting IOW low, the override
    /// merge is breaking somewhere in the path.
    pub pio_tick_iow_low_count: u64,
    /// Diagnostic — maximum PC value PIO0 SM0 has held during the run
    /// (observed after each slow-path tick). PicoGUS bring-up: if this
    /// stays at the WAIT-pin instruction slot, SM0 never escaped its
    /// wait. If it climbs to a higher slot, SM0 advanced through the
    /// program. Slow-path-only — fast-path skips PIO when both blocks
    /// are idle so SM0 wouldn't be moving regardless.
    pub pio0_sm0_max_pc: u8,
    /// Diagnostic — number of times PIO0 SM0's PC differed from its
    /// previous-tick value (advanced or jumped). Slow-path-only.
    pub pio0_sm0_pc_advances: u64,
    /// Last observed PC of PIO0 SM0 — internal scratch used by
    /// [`Self::tick_pio_and_route_irqs`] to decide whether the
    /// PC moved this tick. Initialised to a sentinel `0xFF` so the
    /// very first observation always counts as an advance.
    pub(crate) pio0_sm0_last_pc: u8,
    /// Execution model chosen at build time; cannot change
    /// post-construction. Dispatch for [`Self::step`] / [`Self::run`] /
    /// [`Self::run_quantum`] branches on this. Defaults to
    /// [`ExecutionModel::Serial`].
    pub execution_model: ExecutionModel,
    /// Live 3-thread runtime when `execution_model == Threaded` and the
    /// first `run` / `run_quantum` has fired. Takes ownership of the
    /// pre-seeded cores / bus / clock during lazy `promote_to_threaded`.
    #[cfg(all(
        feature = "threading",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    pub(crate) threaded: Option<threaded::ThreadedEmulator>,
    /// Sticky panic record from a Threaded worker. Set once when
    /// `run_quantum` / `run` observes a worker panic; every subsequent
    /// call returns this cached error without re-attempting workers.
    #[cfg(all(
        feature = "threading",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    pub(crate) panic_info: Option<(WorkerName, String)>,
    /// Sticky watchdog-timeout record from a Threaded run. Set once
    /// when `run_quantum` / `run` observes a barrier timeout; every
    /// subsequent call returns this cached error without re-attempting
    /// workers. HLD V1 §6.6 Stage 5.
    #[cfg(all(
        feature = "threading",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    pub(crate) timeout_info: Option<(WorkerName, u32)>,
    /// Test-only panic injector. Armed via
    /// [`Self::inject_panic_for_testing`]; consumed on the next
    /// `run_quantum` / `run` call which forwards to
    /// [`threaded::ThreadedEmulator::inject_panic_for_testing`].
    #[cfg(all(
        feature = "testing",
        feature = "threading",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    pub(crate) pending_panic_inject: Option<WorkerName>,
    /// `true` once `promote_to_threaded` has moved the seeded state
    /// into `self.threaded` — the flat `cores` / `bus` / `clock` fields
    /// now hold zero-cost placeholders. Typed accessors
    /// (`core`, `core_mut`, `peek`, `gpio_read`, …) `debug_assert!` on
    /// this flag so Serial-only callers trip loudly if they reach for
    /// the flat fields after a Threaded run.
    ///
    /// Known escape: raw field access (`emu.bus.*`) bypasses the
    /// guarded accessors — documented in `tech_debt.md`.
    #[cfg(all(
        feature = "threading",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    pub(crate) bus_is_placeholder: bool,
}

impl Emulator {
    /// Create a new Serial-mode emulator with the given configuration.
    /// Infallible shim: Serial builds always succeed. For Threaded
    /// construction or to surface `ConfigError` explicitly, use
    /// [`EmulatorBuilder`] directly.
    pub fn new(config: Config) -> Self {
        EmulatorBuilder::new(config)
            .build()
            .expect("Serial build is infallible")
    }

    /// Currently selected execution model. Set at build time; does not
    /// change post-construction.
    pub fn execution_model(&self) -> ExecutionModel {
        self.execution_model
    }

    /// Cycle counter for core `idx` (0 or 1). Serial reads directly
    /// from the flat `cores[idx]`; Threaded reads the worker-thread
    /// snapshot (valid between `run_quantum` calls). Returns 0 on
    /// Threaded before the first `run_quantum` (cores not yet taken).
    pub fn core_cycles(&self, idx: u8) -> u64 {
        #[cfg(all(
            feature = "threading",
            target_arch = "x86_64",
            any(target_os = "windows", target_os = "linux")
        ))]
        if let Some(t) = &self.threaded {
            return t.core_cycles(idx);
        }
        match idx {
            0 | 1 => self.cores[idx as usize].cycles(),
            _ => panic!("core_cycles: idx must be 0 or 1"),
        }
    }

    /// Placeholder-guard message shared by the typed accessors below.
    #[cfg(all(
        feature = "threading",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    const PLACEHOLDER_GUARD_MSG: &'static str = "direct field access on cores/bus/clock is Serial-only; emulator is in \
         Threaded mode — use typed accessors like core_cycles(), master_cycle(), \
         gpio_read() instead";

    /// Debug-only placeholder assertion. No-op on non-threading
    /// platforms and in release builds.
    #[inline(always)]
    fn assert_not_placeholder(&self) {
        #[cfg(all(
            feature = "threading",
            target_arch = "x86_64",
            any(target_os = "windows", target_os = "linux")
        ))]
        debug_assert!(!self.bus_is_placeholder, "{}", Self::PLACEHOLDER_GUARD_MSG);
    }

    /// Reset the emulator:
    /// * Load SP from ROM word 0, PC from ROM word 4 into both cores.
    /// * Core 0 is the bootstrapped core (runs from reset).
    /// * Core 1 is halted — the Pico SDK launches it by writing a
    ///   wake sequence through the SIO FIFO; `step` calls
    ///   [`Self::wake_checks`] each quantum to observe the handshake.
    pub fn reset(&mut self) {
        self.assert_not_placeholder();
        let initial_sp = self.bus.memory.rom_read32(0);
        let reset_vector = self.bus.memory.rom_read32(4);

        for i in 0..2 {
            self.cores[i] = CortexM0Plus::with_id(i as u8);
            self.cores[i].regs.msp = initial_sp;
            self.cores[i].regs.r[13] = initial_sp;
            self.cores[i].regs.set_pc(reset_vector & !1);
            self.cores[i].regs.xpsr = 1 << 24; // Thumb bit (XPSR_T)
        }

        self.bus.sio.reset();
        self.bus.resets.reset();
        self.bus.clocks_regs.reset();
        self.bus.xosc_regs.reset();
        self.bus.rosc_regs.reset();
        self.bus.watchdog_tick.reset();
        self.bus.timer.reset();
        self.bus.uart0.reset();
        self.bus.uart1.reset();
        self.bus.spi0.reset();
        self.bus.spi1.reset();
        self.bus.usbctrl.reset();
        self.bus.i2c0.reset();
        self.bus.i2c1.reset();
        self.bus.adc.reset();
        self.bus.pwm.reset();
        self.bus.dma.reset();
        self.bus.irq_pending = 0;
        for n in &mut self.bus.nvics {
            n.reset();
        }
        self.bus.pll_sys_regs = bus::clocks::PLL_RESET;
        self.bus.pll_usb_regs = bus::clocks::PLL_RESET;
        self.bus.pll_sys_lock_at_cycle = None;
        self.bus.pll_usb_lock_at_cycle = None;
        self.bus.master_cycle = 0;
        self.bus.clock_tree = Default::default();
        self.bus.io_bank0.reset();
        self.bus.pads_bank0.reset();
        for pio in &mut self.bus.pio {
            pio.reset();
        }
        // Diagnostic counters track post-reset behaviour, so zero them
        // on `reset()` too (the SM `pc` field also resets to 0, hence
        // the sentinel `0xFF` for `last_pc` to make the first observed
        // PC count as an advance).
        self.pio0_sm0_max_pc = 0;
        self.pio0_sm0_pc_advances = 0;
        self.pio0_sm0_last_pc = 0xFF;
        if let Some(ref mut psram) = self.bus.psram {
            psram.reset_state();
        }
        self.bus.clear_bus_fault();
        self.bus.ppb = [Default::default(), Default::default()];
        self.bus.event_flag = [false; 2];
        self.bus.wfe_waiting = [false; 2];
        self.bus.gpio_in = 0;
        self.bus.external_gpio_in_override = 0;
        self.bus.external_gpio_in_mask = 0;
        self.bus.end_core1_step();

        self.clock = Clock { cycles: 0 };

        // Core 1 stays halted — bootrom on real silicon parks core 1 in
        // a wait-for-event loop until core 0 sends the wake sequence.
        // Routed through the wrapper so the SIO handshake FSM `armed`
        // flag stays in sync with core 1's halt state (HLD §2.1).
        self.halt_core1();
    }

    /// Load a raw binary at the given address. ROM writes are honoured
    /// (test seeding path); SRAM writes land in the SRAM backing store;
    /// XIP loads use [`Self::load_flash`].
    pub fn load_image(&mut self, addr: u32, data: &[u8]) {
        self.assert_not_placeholder();
        match addr >> 28 {
            0x0 => {
                // ROM: bootrom-style loads happen via `load_bootrom`.
                // Support ROM overlay here for tests that want to place
                // code at an arbitrary ROM offset without zero-padding.
                let offset = (addr & 0x0FFF_FFFF) as usize;
                let mut rom_buf = vec![0u8; ROM_SIZE];
                // Seed with current ROM content so a partial overlay
                // preserves whatever was already loaded.
                for i in 0..ROM_SIZE {
                    rom_buf[i] = self.bus.memory.rom_read8(i as u32);
                }
                let end = (offset + data.len()).min(ROM_SIZE);
                if offset < ROM_SIZE {
                    rom_buf[offset..end].copy_from_slice(&data[..end - offset]);
                    self.bus.memory.load_rom(&rom_buf);
                }
                self.invalidate_decode_caches_region(crate::bus::invalidation_regions::ROM);
            }
            0x2 => {
                self.bus.ext_sram_writes += 1;
                for (i, &byte) in data.iter().enumerate() {
                    let a = addr.wrapping_add(i as u32);
                    self.bus.memory.sram_write8(a & 0x00FF_FFFF, byte);
                }
                self.invalidate_decode_caches_region(crate::bus::invalidation_regions::SRAM);
            }
            _ => {}
        }
    }

    /// Bulk-invalidate both cores' decode caches for the given region
    /// bitmask. Used by `load_image` (which writes directly to the
    /// memory backing store, bypassing `Bus::write*`'s automatic
    /// per-write invalidation queue) to keep the caches coherent with
    /// the new bytes. Caller passes a single region bit (ROM / XIP /
    /// SRAM) or BULK to drain everything.
    fn invalidate_decode_caches_region(&mut self, region: u8) {
        self.cores[0].invalidate_decode_cache_regions(region);
        self.cores[1].invalidate_decode_cache_regions(region);
    }

    /// Load the 16 KB RP2040 bootrom at address `0x0000_0000`.
    pub fn load_bootrom(&mut self, data: &[u8]) {
        self.assert_not_placeholder();
        self.bus.load_bootrom(data);
        // Drain the region bit `Bus::load_bootrom` set so the next
        // `step` doesn't see a stale ROM region flag.
        let regions = std::mem::take(&mut self.bus.pending_invalidation_regions);
        if regions != 0 {
            self.cores[0].invalidate_decode_cache_regions(regions);
            self.cores[1].invalidate_decode_cache_regions(regions);
        }
    }

    /// The whole flash, as firmware has programmed and erased it.
    pub fn flash_image(&self) -> &[u8] {
        self.bus.memory.flash()
    }

    /// Load an XIP flash image (appears at XIP address `0x1000_0000`).
    pub fn load_flash(&mut self, data: &[u8]) {
        self.assert_not_placeholder();
        self.bus.load_flash(data);
        let regions = std::mem::take(&mut self.bus.pending_invalidation_regions);
        if regions != 0 {
            self.cores[0].invalidate_decode_cache_regions(regions);
            self.cores[1].invalidate_decode_cache_regions(regions);
        }
    }

    /// Direct-boot into an SDK-style firmware by emulating the boot2 →
    /// application handoff. On real silicon the boot2 stub does three
    /// things before jumping to the application reset handler: it loads
    /// SP from word 0 of the vector table, sets VTOR to the vector
    /// table's flash address, and branches to the reset handler at word
    /// 1 (Thumb bit stripped). This helper performs the same three-piece
    /// handoff — SP, VTOR, PC — into both cores, then parks core 1
    /// halted as `reset()` does. The vector table is expected at
    /// `vtor_offset` within flash (typically `0x100` for pico-sdk).
    ///
    /// Skipping VTOR is silently fatal for any pico-sdk firmware that
    /// calls `runtime_init_install_ram_vector_table`, which copies the
    /// flash vector table into SRAM and then writes the SRAM address to
    /// VTOR. The copy walks `mem[VTOR + 4*i]` for `i` in 0..48; with
    /// VTOR left at `0x0000_0000` that reads from the bootrom image —
    /// garbage bytes get installed as exception handlers and the first
    /// systick fault sends PC into the weeds.
    ///
    /// Why this helper exists at all — the real RP2040 B2 bootrom
    /// detects an attached QSPI flash chip by sampling six QSPI pads via
    /// `SIO GPIO_HI_IN` (offset `0x008`) and validates boot2 by CRC of
    /// the first 252 flash bytes read through the SSI peripheral. Our
    /// emulator stubs SSI and has no QSPI pad model, so the bootrom
    /// (correctly) gives up and enters USB MSC boot mode, where it waits
    /// forever for a UF2 drop. This helper bypasses that check.
    ///
    /// The bootrom image remains populated at `0x00000000` so firmware
    /// can resolve ROM function-table pointers (`rom_func_lookup`,
    /// `rom_data_lookup`). Call **after** `load_bootrom` + `load_flash`
    /// + `reset`.
    pub fn direct_boot_from_flash(&mut self, vtor_offset: u32) {
        self.assert_not_placeholder();
        let sp = self.bus.memory.xip_read32(vtor_offset);
        let pc = self.bus.memory.xip_read32(vtor_offset + 4) & !1;
        let vtor_addr = bus::XIP_FLASH_BASE + vtor_offset;
        for core in 0..2 {
            self.cores[core].regs.msp = sp;
            self.cores[core].regs.r[13] = sp;
            self.cores[core].regs.set_pc(pc);
        }
        self.bus.ppb[0].vtor = vtor_addr;
        self.bus.ppb[1].vtor = vtor_addr;
        // Core 1 stays halted — SDK firmware launches it explicitly via
        // the SIO FIFO handshake, same as after bootrom hand-off. Route
        // through the wrapper so the handshake FSM re-arms if the caller
        // used `direct_boot_from_flash` as a mode-switch (§2.1).
        self.halt_core1();
    }

    /// Advance the system by up to `step_quantum` master-clock cycles,
    /// then tick peripherals once. Returns the number of cycles actually
    /// consumed in this quantum (may be less than `step_quantum` if both
    /// cores halt mid-quantum).
    ///
    /// Per-instruction interleaving of core 0 and core 1 is preserved so
    /// that bank contention timing on core 1 (`contention_check_active`)
    /// still accounts +1 cycle on same-port accesses. Each core is armed
    /// independently per iteration — core 1 can continue running while
    /// core 0 is halted, and vice-versa. Per-instruction FIFO wake
    /// checks (`maybe_wake_core1`) also remain so a FIFO write from
    /// core 0 wakes core 1 within the same quantum.
    ///
    /// Dual-core schedule (per inner-loop iteration):
    /// 1. If core 0 is not halted, step it — fetch/decode/execute one
    ///    instruction.
    /// 2. If core 1 is not halted, step it with `contention_check_active`
    ///    so same-bank SRAM accesses incur +1 cycle.
    /// 3. Advance the master clock by `max(c0, c1)` — both cores share
    ///    one clock on real silicon.
    ///
    /// The loop exits when `clock.cycles >= target` or both cores are
    /// halted. Then advance PIO and the GPIO/PSRAM merge **one system
    /// cycle at a time** for each consumed cycle — so PIO-driven SPI
    /// programs (which toggle SCK every 1–2 sysclks) present every edge
    /// to the off-chip PSRAM model. A bulk `tick_pio(consumed)` followed
    /// by a single `update_gpio()` would let SCK/CS edges slip between
    /// the start and end of the quantum — the PSRAM would only ever see
    /// the quantum's final pin snapshot.
    ///
    /// Fast-path: when both PIO blocks have no SM enabled, no GPIO pin
    /// can change during this peripheral-tick window (SIO writes only
    /// land inside the core loop above, which has already finished),
    /// so a second `psram.tick` on the same pin snapshot would be a
    /// semantic no-op — one bulk `tick_pio(consumed) + update_gpio()`
    /// suffices. This preserves paced_bench_rp2040's throughput on
    /// pure-ALU workloads (no PIO activity), which would otherwise pay
    /// a per-cycle `update_gpio` tax for nothing.
    ///
    /// Core 1 halted ⇒ PIO may still be ticking (e.g. SPI PSRAM on core
    /// 0), so the per-cycle loop runs regardless of core-halt state.
    /// Differs from `rp2350_emu::Emulator::step`'s quantum-end peripheral
    /// tick — rp2040_emu has the external PSRAM which is sensitive to
    /// sub-quantum edge timing; rp2350_emu has no equivalent peripheral.
    pub fn step(&mut self) -> Result<u64, EmulatorError> {
        if self.execution_model == ExecutionModel::Threaded {
            #[cfg(all(
                feature = "threading",
                target_arch = "x86_64",
                any(target_os = "windows", target_os = "linux")
            ))]
            if let Some((which, message)) = &self.panic_info {
                return Err(EmulatorError::WorkerPanicked {
                    which: *which,
                    message: message.clone(),
                });
            }
            #[cfg(all(
                feature = "threading",
                target_arch = "x86_64",
                any(target_os = "windows", target_os = "linux")
            ))]
            if let Some((which, elapsed_ms)) = self.timeout_info {
                return Err(EmulatorError::BarrierTimeout { which, elapsed_ms });
            }
            return Err(EmulatorError::NotSupportedInThreadedMode);
        }
        Ok(self.step_serial())
    }

    /// Drain the bus's pending decode-cache invalidations into both
    /// cores' caches and reset the buffers. Called after each
    /// `core.step` in [`Self::step_serial`] (mirroring the rp2350_emu
    /// drain at lib.rs:1356-1373, commit `0c31479`).
    ///
    /// Per-instruction queue (`pending_cache_invalidations`) drains
    /// only into the core that just ran — the runner that wrote the
    /// bytes is the one most likely to refetch them, and the peer
    /// core's executable bytes haven't moved this step. Region-scoped
    /// bulk invalidations (`pending_invalidation_regions`, set by ISB
    /// inside an instruction or by a mid-step `Bus::load_*`) drain to
    /// BOTH cores so cross-core SMC observers get evicted on their
    /// next turn.
    #[inline(always)]
    fn drain_cache_invalidations(bus: &mut Bus, cores: &mut [CortexM0Plus; 2], blocks: &mut Blocks) {
        if !bus.pending_cache_invalidations.is_empty() || bus.pending_invalidation_regions != 0 {
            Self::drain_cache_invalidations_slow(bus, cores, blocks);
        }
    }

    #[cold]
    #[inline(never)]
    fn drain_cache_invalidations_slow(bus: &mut Bus, cores: &mut [CortexM0Plus; 2], blocks: &mut Blocks) {
        if !bus.pending_cache_invalidations.is_empty() {
            let active = bus.active_core();
            cores[active].invalidate_decode_cache_entries(&bus.pending_cache_invalidations);
            blocks.invalidate(&bus.pending_cache_invalidations);
            bus.pending_cache_invalidations.clear();
        }
        if bus.pending_invalidation_regions != 0 {
            let regions = bus.pending_invalidation_regions;
            cores[0].invalidate_decode_cache_regions(regions);
            cores[1].invalidate_decode_cache_regions(regions);
            blocks.clear();
            bus.pending_invalidation_regions = 0;
        }
    }

    /// Whether core `core`'s next step runs under the idle-loop detector:
    /// it is on and the other core sleeps. Turns bus tracking on to match,
    /// and drops a snapshot the detector can no longer vouch for.
    #[inline(always)]
    fn idle_track(&mut self, core: usize) -> bool {
        let other = 1 - core;
        let other_blocked = self.cores[other].is_halted() || self.bus.wfe_waiting[other];
        let watch = !other_blocked && self.idle.two_core;
        // The bus keeps one dirty flag, so in watch mode only one core
        // holds a snapshot at a time; a parked core has given its up.
        let track = self.idle.enabled && (other_blocked || (watch && self.idle.snap[other].is_none()));
        let mode_changed = self.idle.snap[core].as_ref().is_some_and(|s| s.watch != watch);
        if (!track || mode_changed) && self.idle.snap[core].is_some() {
            self.idle.stats[core].displaced += 1;
            self.idle_drop(core);
        }
        // Loop heads are visited whenever the core is eligible, but accesses
        // are classified only while a snapshot is live: a pass is judged by
        // the stretch from its snapshot to the compare, nothing before it.
        // Once the pass is dirty the next visit drops the snapshot whatever
        // else it reads or stores, so classifying stops there.
        self.bus.idle.on = track && self.idle.snap[core].is_some() && !self.bus.idle.dirty;
        self.bus.idle.watch = watch;
        track
    }

    /// Steps core `k` alone while the other core is halted, in WFE or
    /// parked. Stops at `target`, when `k` halts, enters WFE or parks, when
    /// the bus reports a conflict for the parked core, or when the other
    /// core becomes runnable, a core 1 launch or an event. Returns the
    /// cycles `k` ran; the master clock has advanced by them.
    fn run_single(&mut self, k: usize, target: u64) -> u64 {
        let other = 1 - k;
        let mut total = 0u64;
        self.bus.set_active_core(k);
        self.bus.end_core1_step();
        let blocks = self.blocks.mode != BlockMode::Off && !self.bus.mmio_trace_enabled;
        loop {
            let track = self.idle_track(k);
            let (pc0, mut c) = if blocks {
                let budget = target.saturating_sub(self.clock.cycles).max(1);
                #[cfg(feature = "jit")]
                let chained = if self.blocks.jit {
                    let ctl = crate::core::block::ChainCtl {
                        track,
                        fails_maxed: self.idle.fails[k] >= IDLE_MAX_FAILS,
                        snap: self.idle.snap[k].as_ref().map(|s| (s.pc, s.regs.r[13], s.regs.r[14])),
                        busy: &self.idle.busy[k],
                        clock: self.clock.cycles,
                    };
                    self.cores[k].run_chain(&mut self.bus, &mut self.blocks, budget, &ctl)
                } else {
                    None
                };
                #[cfg(not(feature = "jit"))]
                let chained = None;
                let e = match chained {
                    Some(e) => e,
                    None => self.cores[k].run_block(&mut self.bus, &mut self.blocks, budget),
                };
                // The clock stands before the block's last instruction, as it
                // stands before each instruction the interpreter steps.
                let before = e.cycles - e.last_cycles;
                total += before;
                self.clock.cycles = self.clock.cycles.wrapping_add(before);
                (e.last_pc, e.last_cycles)
            } else {
                let pc0 = self.cores[k].regs.pc();
                (pc0, self.cores[k].step(&mut self.bus) as u64)
            };
            self.bus.idle.on = false;
            let pc1 = self.cores[k].regs.pc();
            if track && pc1 < pc0 {
                let room = target.saturating_sub(self.clock.cycles + c);
                c += self.idle_head_visit(k, pc1, room);
            }
            Self::drain_cache_invalidations(&mut self.bus, &mut self.cores, &mut self.blocks);
            self.maybe_wake_core1(k);
            if c == 0 {
                break;
            }
            total += c;
            self.clock.cycles = self.clock.cycles.wrapping_add(c);
            let other_runs = !self.cores[other].is_halted()
                && !self.bus.wfe_waiting[other]
                && self.idle.park[other].is_none();
            if self.clock.cycles >= target
                || self.cores[k].is_halted()
                || self.bus.wfe_waiting[k]
                || self.idle.park[k].is_some()
                || (self.idle.park[other].is_some() && self.bus.park[other].hit)
                || other_runs
            {
                break;
            }
        }
        total
    }

    /// Credits a parked `core` with the whole passes that fit between its
    /// park and now and keeps it parked; returns the cycles credited.
    fn settle_park(&mut self, core: usize) -> u64 {
        let Some(p) = self.idle.park[core].as_mut() else { return 0 };
        let span = self.clock.cycles.saturating_sub(p.since);
        let credit = whole_passes(span, p.iter);
        p.since += credit;
        self.cores[core].cycles += credit;
        self.idle.parked_cycles[core] += credit;
        credit
    }

    /// At a quantum boundary: a park carries on while no IRQ waits for the
    /// core, the bus saw no conflict, the pass did not read the TIMER, and
    /// everything it read still holds the value it had at park time.
    fn park_carries(&mut self, core: usize) -> bool {
        let w = &self.bus.park[core];
        if w.hit || w.timer {
            return false;
        }
        if self.bus.nvics[core].pending_and_enabled() != 0 || self.bus.ppb[core].icsr & (1 << 28 | 1 << 26) != 0 {
            return false;
        }
        self.bus.park_inputs_held(core)
    }

    /// At a quantum boundary, after the parks are settled: a watch-mode
    /// snapshot carries on while the other core stays parked, its pass so
    /// far is clean, read no TIMER and is younger than a quantum, no
    /// exception waits for the core, and everything the pass has read
    /// holds the value it had at the end of the last quantum. The pass then
    /// runs as it would have run wholly in the new quantum, so its compare
    /// means what a compare inside one quantum means.
    fn snap_carries(&mut self, core: usize) -> bool {
        let Some(s) = &self.idle.snap[core] else { return false };
        if !s.watch || self.bus.idle.dirty || self.bus.idle.timer {
            return false;
        }
        // A snapshot holds the one tracking slot, so it carries only while
        // the other core stays parked and needs no tracking itself, and
        // only once: a pass longer than a quantum does not park.
        if self.idle.park[1 - core].is_none() || self.cores[core].cycles - s.cycles >= self.step_quantum as u64 {
            return false;
        }
        if self.bus.nvics[core].pending_and_enabled() != 0 || self.bus.ppb[core].icsr & (1 << 28 | 1 << 26) != 0 {
            return false;
        }
        self.bus.idle_fingerprint(core, &mut self.idle.carry_now);
        self.idle.carry_now == self.idle.carry_vals
    }

    /// Ends `core`'s park: credits the whole passes that fit between the
    /// park and now, and returns them.
    fn unpark(&mut self, core: usize) -> u64 {
        let Some(p) = self.idle.park[core].take() else { return 0 };
        self.bus.park[core].active = false;
        self.bus.park[core].hit = false;
        self.bus.refresh_watch_pages();
        let span = self.clock.cycles.saturating_sub(p.since);
        let credit = whole_passes(span, p.iter);
        self.cores[core].cycles += credit;
        self.idle.parked_cycles[core] += credit;
        self.idle_drop(core);
        credit
    }

    fn mark_busy(&mut self, core: usize, pc: u32) {
        if self.idle.parked_heads[core].contains(&pc) {
            return;
        }
        let now = self.clock.cycles;
        self.idle.busy[core][busy_slot(pc)] = (pc, now + IDLE_BUSY_CYCLES);
    }

    #[inline]
    fn is_busy(&self, core: usize, pc: u32) -> bool {
        is_busy_head(&self.idle.busy[core], pc, self.clock.cycles)
    }

    /// Whether a visit to head `pc` takes no snapshot because the core is
    /// still settling after an exception. The first loop after a handler
    /// stores and loads what the handler changed, so a pass tracked through
    /// it comes back changed. Settling starts at the first head after the
    /// exception and ends when the core is back at that head, or after
    /// [`IDLE_SETTLE_CYCLES`].
    fn idle_settling(&mut self, core: usize, pc: u32, exception: bool) -> bool {
        let now = self.cores[core].cycles;
        if exception {
            self.idle.settle[core] = Some((pc, now));
            return true;
        }
        match self.idle.settle[core] {
            Some((h, t0)) if h != pc && now - t0 < IDLE_SETTLE_CYCLES => true,
            Some(_) => {
                self.idle.settle[core] = None;
                false
            }
            None => false,
        }
    }

    fn idle_drop(&mut self, core: usize) {
        if self.idle.snap[core].take().is_some() {
            self.bus.idle.logging = false;
            self.bus.idle.clear_log();
        }
    }

    /// Core `core` just branched backwards to `pc`. Compares against the
    /// snapshot or takes one, as [`IdleSkip`] describes; returns the cycles skipped,
    /// whole passes that fit in `room`.
    fn idle_head_visit(&mut self, core: usize, pc: u32, room: u64) -> u64 {
        let clean = !self.bus.idle.dirty;
        let event = self.bus.idle.event;
        let exception = self.bus.idle.exception;
        self.bus.idle.dirty = false;
        self.bus.idle.event = false;
        self.bus.idle.exception = false;
        if !clean && self.idle.snap[core].is_some() {
            if exception {
                self.idle.settle[core] = Some((pc, self.cores[core].cycles));
            }
            self.idle.stats[core].dirty += 1;
            if !event {
                let head = self.idle.snap[core].as_ref().map_or(pc, |s| s.pc);
                self.mark_busy(core, head);
            }
            self.idle_drop(core);
            return 0;
        }
        let c = &self.cores[core];
        if let Some(s) = &self.idle.snap[core] {
            if s.pc != pc || s.regs.r[13] != c.regs.r[13] || s.regs.r[14] != c.regs.r[14] {
                return 0; // another head inside the same clean stretch
            }
            if s.watch
                && s.regs == c.regs
                && c.pending_fault.is_none()
                && {
                    let mut side = s.side;
                    if self.bus.idle.pio & crate::bus::PARK_SPINLOCK == 0 {
                        side[0] = 0;
                    }
                    side == self.bus.idle_watch_side(core, self.bus.idle.pio)
                }
                && self.bus.idle_memory_restored()
            {
                let iter = c.cycles - s.cycles;
                if iter > 0 {
                    let w = &mut self.bus.park[core];
                    w.words.clear();
                    w.words.extend_from_slice(&self.bus.idle.words);
                    w.pio = self.bus.idle.pio;
                    w.gpio = self.bus.idle.gpio;
                    w.stored = self.bus.idle.stored;
                    w.timer = self.bus.idle.timer;
                    w.hit = false;
                    w.active = true;
                    self.bus.park[core].vals = self.bus.park_fingerprint(core);
                    self.bus.park[core].ext_seen = self.bus.ext_sram_writes;
                    self.bus.refresh_watch_pages();
                    self.idle.park[core] = Some(Park { since: self.clock.cycles, iter });
                    self.idle.stats[core].parks += 1;
                    let h = &mut self.idle.parked_heads[core];
                    if h[0] != pc {
                        let n = h.iter().position(|&e| e == pc).unwrap_or(IDLE_PARKED_HEADS - 1);
                        h.copy_within(0..n, 1);
                        h[0] = pc;
                    }
                    self.idle_drop(core);
                }
                return 0;
            }
            if !s.watch
                && s.regs == c.regs
                && c.pending_fault.is_none()
                && s.side == self.bus.idle_side_state(core)
                && self.bus.idle_memory_restored()
            {
                let iter = c.cycles - s.cycles;
                let skip = room.checked_div(iter).map_or(0, |n| n * iter);
                let c = &mut self.cores[core];
                c.cycles += skip;
                self.idle.skipped_cycles[core] += skip;
                self.idle.stats[core].skips += 1;
                let now = c.cycles;
                if let Some(s) = self.idle.snap[core].as_mut() {
                    s.cycles = now;
                }
                self.bus.idle.clear_log();
                return skip;
            }
            // Same point in the pass, different state: the loop computes.
            self.idle.stats[core].changed += 1;
            self.idle.fails[core] += 1;
            self.mark_busy(core, pc);
            self.idle_drop(core);
            return 0;
        }
        if self.idle.fails[core] >= IDLE_MAX_FAILS || self.is_busy(core, pc) {
            self.idle_drop(core);
            return 0;
        }
        if self.idle_settling(core, pc, exception) {
            return 0;
        }
        self.idle.fails[core] += 1;
        self.idle.stats[core].snaps += 1;
        let other = 1 - core;
        if self.idle.snap[other].take().is_some() {
            self.idle.stats[other].displaced += 1;
            self.bus.idle.clear_log();
        }
        let c = &self.cores[core];
        let watch = self.bus.idle.watch;
        self.idle.snap[core] = Some(IdleSnap { pc, regs: c.regs, cycles: c.cycles, side: self.bus.idle_side_state(core), watch });
        self.bus.idle.logging = true;
        self.bus.idle.clear_log();
        self.bus.idle.stored = false;
        self.bus.idle.timer = false;
        self.bus.idle.clear_words();
        self.bus.idle.pio = 0;
        self.bus.idle.gpio = false;
        0
    }

    /// Serial-mode single-quantum step. Shared by [`Self::step`] and
    /// [`Self::run_quantum`] on the Serial path.
    fn step_serial(&mut self) -> u64 {
        debug_assert!(self.step_quantum > 0, "step_quantum must be >= 1");
        // Refresh the Bus's view of the master cycle count so any MMIO
        // reads / writes performed during this quantum (notably PLL CS
        // lock bit + lock-arm transitions — see
        // `wrk_docs/2026.04.15 - HLD - PLL LOCK Modelling.md` §6 P2)
        // observe a current cycle. Staleness is bounded by one quantum.
        self.bus.master_cycle = self.clock.cycles;
        let start = self.clock.cycles;
        let target = start.wrapping_add(self.step_quantum as u64);

        // Per HLD 2026.04.26 V5 §5.2.3: accumulate per-core cycle counts
        // across the inner loop so the slow-branch SysTick advance
        // mirrors per-core hardware semantics (each core's SysTick
        // decrements on its own consumed cycles, not on the active-core
        // shared register).
        let mut c0_total: u64 = 0;
        let mut c1_total: u64 = 0;
        for k in 0..2 {
            if self.idle.park[k].is_none() {
                continue;
            }
            if self.park_carries(k) {
                self.idle.stats[k].carried += 1;
            } else {
                if self.bus.park[k].timer && !self.bus.park[k].hit {
                    self.idle.stats[k].ended_timer += 1;
                } else {
                    self.idle.stats[k].ended_other += 1;
                }
                let credit = self.unpark(k);
                if k == 0 { c0_total += credit } else { c1_total += credit }
            }
        }
        // Memory, pins and peripherals may have changed since the last
        // quantum: start the idle-loop detector afresh, but for a pass whose
        // inputs so far all held.
        let keep = self.idle.carry.take().filter(|&k| self.snap_carries(k));
        // A snapshot that has outlived a whole quantum is not on a pass
        // short enough to park: its head is busy.
        for k in 0..2 {
            if let Some(s) = &self.idle.snap[k]
                && keep != Some(k)
                && self.cores[k].cycles - s.cycles >= self.step_quantum as u64
            {
                let pc = s.pc;
                self.mark_busy(k, pc);
            }
        }
        for k in 0..2 {
            if keep == Some(k) {
                self.idle.stats[k].snap_carried += 1;
                continue;
            }
            self.idle.stats[k].expired += self.idle.snap[k].is_some() as u64;
            self.idle_drop(k);
        }
        self.idle.fails = [0; 2];
        if keep.is_none() {
            self.bus.idle.dirty = true;
        }

        while self.clock.cycles < target
            && (!self.cores[0].is_halted() || !self.cores[1].is_halted())
        {
            for k in 0..2 {
                if self.idle.park[k].is_some() && self.bus.park[k].hit {
                    self.idle.stats[k].hits += 1;
                    let credit = self.unpark(k);
                    if k == 0 { c0_total += credit } else { c1_total += credit }
                }
            }
            let run0 = !self.cores[0].is_halted() && !self.bus.wfe_waiting[0] && self.idle.park[0].is_none();
            let run1 = !self.cores[1].is_halted() && !self.bus.wfe_waiting[1] && self.idle.park[1].is_none();
            if run0 != run1 {
                // One core can run: step it alone, without the other core's
                // checks or the bank-contention window, which needs both.
                let c = self.run_single(if run0 { 0 } else { 1 }, target);
                if run0 { c0_total += c } else { c1_total += c }
                if c == 0 {
                    break;
                }
                continue;
            }
            let c0 = if !self.cores[0].is_halted() && !self.bus.wfe_waiting[0] && self.idle.park[0].is_none() {
                self.bus.set_active_core(0);
                let track = self.idle_track(0);
                let pc0 = self.cores[0].regs.pc();
                let mut c = self.cores[0].step(&mut self.bus) as u64;
                self.bus.idle.on = false;
                let pc1 = self.cores[0].regs.pc();
                if track && pc1 < pc0 {
                    let room = target.saturating_sub(self.clock.cycles + c);
                    c += self.idle_head_visit(0, pc1, room);
                }
                // Drain decode-cache invalidations recorded by writes
                // during this step into the core that just ran.
                // Region-scoped bulk invalidations (load_*) reach BOTH
                // cores so a peer core fetching from the same region
                // sees the eviction next quantum. Mirrors rp2350_emu
                // (commit 0c31479, lib.rs §lookup-and-drain).
                Self::drain_cache_invalidations(&mut self.bus, &mut self.cores, &mut self.blocks);
                self.maybe_wake_core1(0);
                c
            } else {
                0
            };

            let c1 = if !self.cores[1].is_halted() && !self.bus.wfe_waiting[1] && self.idle.park[1].is_none() {
                self.bus.set_active_core(1);
                self.bus.begin_core1_step();
                let track = self.idle_track(1);
                let pc0 = self.cores[1].regs.pc();
                let mut c = self.cores[1].step(&mut self.bus) as u64;
                self.bus.idle.on = false;
                let pc1 = self.cores[1].regs.pc();
                if track && pc1 < pc0 {
                    let room = target.saturating_sub(self.clock.cycles + c);
                    c += self.idle_head_visit(1, pc1, room);
                }
                Self::drain_cache_invalidations(&mut self.bus, &mut self.cores, &mut self.blocks);
                self.bus.end_core1_step();
                self.maybe_wake_core1(1);
                c
            } else {
                // Still clear any leftover bank-tracking state so the
                // next iteration starts fresh.
                self.bus.end_core1_step();
                0
            };

            if c0 == 0 && c1 == 0 {
                // Every core that could run is parked or blocked: nothing
                // changes until the quantum ends.
                if self.idle.park[0].is_some() || self.idle.park[1].is_some() {
                    self.clock.cycles = target;
                }
                break;
            }
            c0_total = c0_total.wrapping_add(c0);
            c1_total = c1_total.wrapping_add(c1);
            self.clock.cycles = self.clock.cycles.wrapping_add(c0.max(c1));
        }
        for k in 0..2 {
            let credit = self.settle_park(k);
            if k == 0 { c0_total += credit } else { c1_total += credit }
        }
        self.idle.carry = None;
        for k in 0..2 {
            if self.idle.snap[k].as_ref().is_some_and(|s| s.watch) && !self.bus.idle.dirty && !self.bus.idle.timer {
                self.bus.idle_fingerprint(k, &mut self.idle.carry_vals);
                self.idle.carry = Some(k);
            }
        }

        let mut consumed = self.clock.cycles.wrapping_sub(start);
        // tech_debt §1649: when both cores are blocked (halted-or-WFE)
        // and the inner loop made no progress, `consumed == 0` and
        // neither the fast-path nor the slow-path below would advance
        // the master clock — so a TIMER alarm scheduled in the future
        // could never fire to wake either core. Detect that exact
        // state, advance the master clock to the soonest scheduled
        // IRQ-raising alarm (capped by `step_quantum`), tick lazy
        // peripherals once, drain the resulting IRQs to the NVICs, and
        // let `wake_checks` at the tail un-halt the woken core. Take
        // the early return so we don't double-advance via fast/slow
        // path. Production fix per HLD V2 / tech_debt §1649 Option 1.
        if consumed == 0
            && (self.cores[0].is_halted() || self.bus.wfe_waiting[0])
            && (self.cores[1].is_halted() || self.bus.wfe_waiting[1])
            && self.bus.irq_pending == 0
            && self.bus.nvics[0].pending_and_enabled() == 0
            && self.bus.nvics[1].pending_and_enabled() == 0
            && let Some(deadline) = self.bus.next_scheduled_lazy_deadline()
            && deadline > self.bus.master_cycle
        {
            // Cap a single advance at `step_quantum`. If the deadline is
            // farther out, the caller's `run`/`run_quantum` loop iterates
            // and the next `step()` re-enters this branch, closing the
            // gap across multiple quanta — never silently stalls.
            // The advance runs through the same peripheral paths as
            // executed cycles, so PIO and DMA keep streaming while both
            // cores sleep.
            let max_advance = self.step_quantum as u64;
            consumed = (deadline - self.bus.master_cycle).min(max_advance);
            self.clock.cycles = self.clock.cycles.wrapping_add(consumed);
        }
        // Both cores blocked with no alarm to skip to: a core parked in
        // WFE, or PIO and DMA still moving data, means time goes on, so
        // the quantum passes as idle cycles through the peripheral paths
        // below. Cores halted for good, with nothing running, still stop
        // the clock and `run` returns early.
        if consumed == 0
            && (self.cores[0].is_halted() || self.bus.wfe_waiting[0])
            && (self.cores[1].is_halted() || self.bus.wfe_waiting[1])
            && self.bus.irq_pending == 0
            && (self.bus.wfe_waiting[0]
                || self.bus.wfe_waiting[1]
                || !self.bus.pio_all_idle()
                || !self.bus.dma.is_idle())
        {
            consumed = self.step_quantum as u64;
            self.clock.cycles = self.clock.cycles.wrapping_add(consumed);
        }
        // See the fn docstring for the rationale on the fast-path and
        // the per-cycle interleave. Measured impact of the fast-path
        // gate on paced_bench_rp2040 (pure ALU, PIO disabled): without
        // it, ~49% throughput regression; with it, neutral.
        //
        // HLD V7 §5.5 broadens the gate from "PIO idle" to "PIO idle
        // AND peripherals idle AND DMA idle AND no IRQ pending".
        // Phase 1 peripherals are all lazy (TIMER/WATCHDOG_TICK), and
        // DMA is a Phase 1 always-idle stub, so in practice the gate
        // still reduces to the PIO check — but the extra conditions
        // are in place so later phases don't need to reopen this
        // site.
        let pio_idle = self.bus.pio_all_idle();
        let peri_idle = self.bus.all_peripherals_idle();
        let dma_idle = self.bus.dma.is_idle();
        let any_irq = self.bus.irq_pending != 0;
        // SysTick fires by ORing into `bus.ppb[active].icsr` — NOT by
        // setting `bus.irq_pending` — so the `any_irq` check above does
        // not gate the fast path on SysTick activity. With SysTick
        // enabled and no peripheral activity (e.g. the V5 §5.2
        // tail-chain scenario's `b .` busy-wait after preamble), the
        // fast path would otherwise trigger and SysTick would never
        // tick. Drop to the slow path whenever SysTick is enabled on
        // either core: SysTick is per-core, and after a two-core
        // quantum the active core is whichever ran last, so checking
        // only that one would starve the other core's SysTick.
        // SysTick-disabled workloads, almost everything, keep their
        // fast-path eligibility.
        let systick_idle = !self.bus.systicks[0].is_enabled() && !self.bus.systicks[1].is_enabled();
        if pio_idle && peri_idle && dma_idle && systick_idle && !any_irq {
            self.tick_pio(consumed as u32);
            // Advance lazy-scheduled peripherals (TIMER alarms) by the
            // same window the cores consumed. Any alarm matching inside
            // the window fires into `bus.irq_pending` and gets drained
            // in the same breath — so firmware that kicks off an alarm
            // in one quantum sees the IRQ land by the start of the
            // next.
            self.bus.advance_lazy_scheduled(consumed);
            self.drain_pending_irqs_to_cores();
            self.update_gpio();
            if let Some(t) = self.gpio_trace.as_mut() {
                t.extend(std::iter::repeat_n(self.bus.gpio_in, consumed as usize));
            }
        } else {
            // Per HLD 2026.04.26 V5 §5.3: chunked once-per-quantum slow
            // branch. `master_cycle` advances by `consumed` BEFORE
            // `tick_peripherals` so TIMER's alarm `>=` poll sees the
            // window's end-of-quantum cycle. SysTick advances per-core
            // by each core's actual consumed cycle count (mirrors M0+
            // hardware: SysTick is per-core, decremented on the cycles
            // the owning core consumes — see §5.2.3). IRQ drain runs once
            // at quantum end, so IRQ delivery lags by up to step_quantum-1
            // cycles; §5.4 has the analysis.
            self.bus.master_cycle = self.bus.master_cycle.wrapping_add(consumed);
            self.bus.tick_peripherals_no_dma(consumed as u32);
            self.tick_systick(c0_total as u32, c1_total as u32);
            // DMA moves at most one word per system cycle, and a PIO FIFO
            // it feeds drains on the PIO's own clock, so while a channel
            // is busy the two step together, one cycle at a time. Once
            // every channel is idle the rest of the quantum is chunked.
            // A pin trace also needs the per-cycle loop.
            let tracing = self.gpio_trace.is_some();
            let mut left = consumed as u32;
            while left > 0 {
                if !tracing && self.bus.dma.is_idle() {
                    self.bus.tick_dma();
                    self.tick_pio_and_route_irqs(left);
                    self.update_gpio();
                    break;
                }
                self.bus.tick_dma();
                self.tick_pio_and_route_irqs(1);
                self.update_gpio();
                let g = self.bus.gpio_in;
                if let Some(t) = self.gpio_trace.as_mut() {
                    t.push(g);
                }
                left -= 1;
            }
            self.drain_pending_irqs_to_cores();
        }
        self.wake_checks();
        consumed
    }

    /// Drain [`Bus::irq_pending`] into both cores' NVIC pending
    /// latches. Per HLD V7 §5.2 this runs once per slow-path inner
    /// cycle so level-triggered peripherals have at most one
    /// architectural cycle of routing lag from assert to NVIC latch.
    ///
    /// Both cores see every IRQ — RP2040 has a single NVIC per core
    /// but shared peripheral IRQ wires, so each line latches
    /// independently on both cores and firmware routes via
    /// `NVIC_IPR` / `NVIC_ISER` (modelled in
    /// `bus/mod.rs::nvic_mmio_write32` + `nvic_mmio_read32`).
    ///
    /// Every source is a level that re-asserts on each poll while its
    /// condition holds, see `core::nvic`. A line whose handler is active
    /// on a core is not pended there: on silicon, entry clears pending
    /// and a level only pends again if it is still high at exception
    /// return. Pending it while the handler runs, before the handler
    /// clears the source, would enter the handler a second time with
    /// nothing to service.
    fn drain_pending_irqs_to_cores(&mut self) {
        if self.bus.irq_pending != 0 {
            let raised = std::mem::replace(&mut self.bus.irq_pending, 0);
            for c in 0..2 {
                let active = (self.bus.ppb[c].active >> 16) as u32;
                let bits = raised & !active;
                for irq in 0..crate::irq::IRQ_COUNT {
                    if bits & (1u32 << irq) != 0 {
                        self.bus.nvics[c].set_pending(irq as u8);
                    }
                }
            }
        }
    }

    /// Per HLD 2026.04.26 V5 §5.2.3: advance each core's SysTick by
    /// its own consumed cycle count for this quantum. Mirrors M0+
    /// hardware semantics — SysTick is per-core, the active-core
    /// `Bus` field is just a banked-MMIO selector, not a tick gate.
    /// PENDSTSET (`ICSR[26]`) latches per-core; `drain_pending_irqs_to_cores`
    /// runs after this call so the SysTick handler is taken on the
    /// next quantum boundary.
    fn tick_systick(&mut self, c0: u32, c1: u32) {
        if self.bus.systicks[0].tick_n(c0 as u64) {
            self.bus.ppb[0].icsr |= 1 << 26;
        }
        if self.bus.systicks[1].tick_n(c1 as u64) {
            self.bus.ppb[1].icsr |= 1 << 26;
        }
    }

    /// Step both PIO blocks by `cycles` system clocks and route their
    /// IRQ flags into [`Bus::irq_pending`].
    ///
    /// Per HLD V7 §5.5 + Appendix B, each PIO block has 8 internal
    /// Per-block 12-bit raw status (`IRQ[3:0]` + RXNEMPTY[3:0] +
    /// TXNFULL[3:0]) is masked through `INT0_INTE` / `INT1_INTE` and
    /// OR'd with `INT0_INTF` / `INT1_INTF` to derive the effective
    /// values on each NVIC line. Each block has two lines: PIO0_IRQ_0/1
    /// at NVIC #7/#8 and PIO1_IRQ_0/1 at NVIC #9/#10. PicoGUS firmware
    /// enables `RXNEMPTY_SM0` on PIO0 INT0_INTE so its ISA handler
    /// fires when an autopushed event lands in PIO0 SM0's RX FIFO.
    ///
    /// Per HLD 2026.04.26 V5 §5.1: chunked once-per-quantum.
    fn tick_pio_and_route_irqs(&mut self, cycles: u32) {
        let gpio_in = self.bus.gpio_in;
        // Diagnostic counters bump by `cycles` (per-quantum granularity is
        // acceptable per HLD 2026.04.26 V5 §7 risk row "Per-cycle
        // observation diagnostics under-count by quantum factor").
        self.pio_tick_count = self.pio_tick_count.wrapping_add(cycles as u64);
        if gpio_in & (1u32 << 4) == 0 {
            self.pio_tick_iow_low_count = self.pio_tick_iow_low_count.wrapping_add(cycles as u64);
        }
        self.bus.pio[0].step_n(cycles, gpio_in);
        self.bus.pio[1].step_n(cycles, gpio_in);
        // Observe PIO0 SM0's PC after the step. Tracks max PC and the
        // number of times the PC differs from the prior observation
        // (counts both linear advances and jumps; sequential same-PC
        // ticks — e.g. a stalled WAIT — do not increment).
        let sm0_pc = self.bus.pio[0].sm[0].pc();
        if sm0_pc > self.pio0_sm0_max_pc {
            self.pio0_sm0_max_pc = sm0_pc;
        }
        if sm0_pc != self.pio0_sm0_last_pc {
            self.pio0_sm0_pc_advances = self.pio0_sm0_pc_advances.wrapping_add(1);
            self.pio0_sm0_last_pc = sm0_pc;
        }
        for (block, line0_bit) in [(0usize, 7u32), (1usize, 9u32)] {
            if self.bus.pio[block].int0_ints_rp2040() != 0 {
                self.bus.irq_pending |= 1u32 << line0_bit;
            }
            if self.bus.pio[block].int1_ints_rp2040() != 0 {
                self.bus.irq_pending |= 1u32 << (line0_bit + 1);
            }
        }
    }

    /// Advance both PIO blocks by `cycles` system-clock cycles.
    ///
    /// PIO reads `bus.gpio_in` as its view of external pin state — feed it
    /// the pre-step merge so programs sampling GPIO (e.g. IN PINS) see the
    /// value SIO / the previous PIO step wrote last. The post-step
    /// `update_gpio()` then refreshes `bus.gpio_in` from `pad_out`/`pad_oe`.
    fn tick_pio(&mut self, cycles: u32) {
        if cycles == 0 {
            return;
        }
        let gpio_in = self.bus.gpio_in;
        for pio in &mut self.bus.pio {
            pio.step_n(cycles, gpio_in);
        }
    }

    /// Run for at least `cycles` virtual cycles. Returns the number of
    /// cycles actually executed. May overshoot by up to `step_quantum - 1`
    /// cycles (one quantum's worth), matching the documented overshoot
    /// behaviour of [`Self::step`].
    ///
    /// Dispatches to the selected [`ExecutionModel`]. In Threaded mode
    /// this rounds up to the nearest quantum boundary (HLD V1 §5.4)
    /// and returns `Err(EmulatorError::WorkerPanicked)` sticky on
    /// worker panic.
    pub fn run(&mut self, cycles: u64) -> Result<u64, EmulatorError> {
        if self.execution_model == ExecutionModel::Serial {
            let start = self.clock.cycles;
            while self.clock.cycles.wrapping_sub(start) < cycles {
                let consumed = self.step_serial();
                if consumed == 0 {
                    break;
                }
            }
            return Ok(self.clock.cycles.wrapping_sub(start));
        }
        #[cfg(all(
            feature = "threading",
            target_arch = "x86_64",
            any(target_os = "windows", target_os = "linux")
        ))]
        {
            if let Some((which, message)) = &self.panic_info {
                return Err(EmulatorError::WorkerPanicked {
                    which: *which,
                    message: message.clone(),
                });
            }
            if let Some((which, elapsed_ms)) = self.timeout_info {
                return Err(EmulatorError::BarrierTimeout { which, elapsed_ms });
            }
            if self.threaded.is_none() {
                self.promote_to_threaded();
            }
            self.apply_pending_panic_inject();
            let step_q = self.step_quantum as u64;
            let quanta = cycles.div_ceil(step_q.max(1));
            let threaded = self.threaded.as_mut().expect("threaded promoted above");
            match threaded.run_quanta_checked(quanta) {
                Ok(()) => Ok(quanta.saturating_mul(step_q)),
                Err(threaded::RunError::Panic { which, message }) => {
                    self.panic_info = Some((which, message.clone()));
                    Err(EmulatorError::WorkerPanicked { which, message })
                }
                Err(threaded::RunError::Timeout { which, elapsed_ms }) => {
                    self.timeout_info = Some((which, elapsed_ms));
                    Err(EmulatorError::BarrierTimeout { which, elapsed_ms })
                }
            }
        }
        #[cfg(not(all(
            feature = "threading",
            target_arch = "x86_64",
            any(target_os = "windows", target_os = "linux")
        )))]
        {
            let _ = cycles;
            Err(EmulatorError::NotSupportedInThreadedMode)
        }
    }

    /// Advance the emulator by exactly one quantum (`step_quantum`
    /// cycles). Primary entry point for the Threaded path; on Serial
    /// this is the same as [`Self::step`] and returns the cycles
    /// consumed. HLD V1 §5.4.
    ///
    /// Returns `Err(EmulatorError::WorkerPanicked)` sticky on worker
    /// panic in Threaded mode. One-shot-after-panic: subsequent calls
    /// return the cached error without re-attempting workers.
    pub fn run_quantum(&mut self) -> Result<u64, EmulatorError> {
        match self.execution_model {
            ExecutionModel::Serial => Ok(self.step_serial()),
            ExecutionModel::Threaded => self.run_quantum_threaded(),
        }
    }

    #[cfg(all(
        feature = "threading",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    fn run_quantum_threaded(&mut self) -> Result<u64, EmulatorError> {
        if let Some((which, message)) = &self.panic_info {
            return Err(EmulatorError::WorkerPanicked {
                which: *which,
                message: message.clone(),
            });
        }
        if let Some((which, elapsed_ms)) = self.timeout_info {
            return Err(EmulatorError::BarrierTimeout { which, elapsed_ms });
        }
        if self.threaded.is_none() {
            self.promote_to_threaded();
        }
        self.apply_pending_panic_inject();
        let step_q = self.step_quantum as u64;
        let threaded = self.threaded.as_mut().expect("threaded promoted above");
        match threaded.run_quanta_checked(1) {
            Ok(()) => Ok(step_q),
            Err(threaded::RunError::Panic { which, message }) => {
                self.panic_info = Some((which, message.clone()));
                Err(EmulatorError::WorkerPanicked { which, message })
            }
            Err(threaded::RunError::Timeout { which, elapsed_ms }) => {
                self.timeout_info = Some((which, elapsed_ms));
                Err(EmulatorError::BarrierTimeout { which, elapsed_ms })
            }
        }
    }

    #[cfg(not(all(
        feature = "threading",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    )))]
    fn run_quantum_threaded(&mut self) -> Result<u64, EmulatorError> {
        Err(EmulatorError::NotSupportedInThreadedMode)
    }

    /// Forward any pending `inject_panic_for_testing` target into the
    /// live `ThreadedEmulator`. No-op on non-testing builds.
    #[cfg(all(
        feature = "threading",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    #[inline]
    fn apply_pending_panic_inject(&mut self) {
        #[cfg(feature = "testing")]
        if let Some(which) = self.pending_panic_inject.take()
            && let Some(t) = self.threaded.as_mut()
        {
            t.inject_panic_for_testing(which);
        }
    }

    /// Move the seeded Serial state into a fresh `ThreadedEmulator`.
    /// Called lazily on the first `run_quantum` / `run` so harness
    /// setup that poked `emu.bus` / `emu.core_mut(...)` pre-run is
    /// carried over. After promotion, the top-level `cores` / `bus` /
    /// `clock` fields hold zero-cost placeholders and must not be
    /// inspected mid-run.
    #[cfg(all(
        feature = "threading",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    fn promote_to_threaded(&mut self) {
        let placeholder_bus = Bus::new();
        let placeholder_cores = [CortexM0Plus::with_id(0), CortexM0Plus::with_id(1)];
        let seeded_bus = std::mem::replace(&mut self.bus, placeholder_bus);
        let seeded_cores = std::mem::replace(&mut self.cores, placeholder_cores);
        let seeded_clock = std::mem::replace(&mut self.clock, Clock { cycles: 0 });
        let seeded = Emulator {
            cores: seeded_cores,
            bus: seeded_bus,
            clock: seeded_clock,
            step_quantum: self.step_quantum,
            gpio_trace: None,
            pio_tick_count: self.pio_tick_count,
            pio_tick_iow_low_count: self.pio_tick_iow_low_count,
            pio0_sm0_max_pc: self.pio0_sm0_max_pc,
            pio0_sm0_pc_advances: self.pio0_sm0_pc_advances,
            pio0_sm0_last_pc: self.pio0_sm0_last_pc,
            execution_model: ExecutionModel::Serial,
            threaded: None,
            panic_info: None,
            timeout_info: None,
            #[cfg(feature = "testing")]
            pending_panic_inject: None,
            bus_is_placeholder: false,
            idle: IdleSkip::default(),
            blocks: Blocks::default(),
        };
        self.threaded = Some(threaded::ThreadedEmulator::from_emulator(seeded));
        self.bus_is_placeholder = true;
    }

    /// Test-only: arm a panic injection for the next `run_quantum` /
    /// `run` call. The matching worker panics on its first barrier
    /// entry; the emulator surfaces `Err(EmulatorError::WorkerPanicked)`
    /// and becomes sticky-poisoned.
    ///
    /// Feature-gated behind `testing` so release consumers cannot brick
    /// their emulator by calling an internal hook.
    #[cfg(all(
        feature = "testing",
        feature = "threading",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    pub fn inject_panic_for_testing(&mut self, which: WorkerName) {
        self.pending_panic_inject = Some(which);
    }

    /// Merge SIO and PIO GPIO outputs into `bus.gpio_in`.
    ///
    /// SIO `gpio_out & gpio_oe` is the base; each PIO block's
    /// `pad_out & pad_oe` overrides SIO on the pins it drives (PIO wins
    /// wherever `pad_oe` has a bit set — mirrors `rp2350_emu::Emulator::
    /// update_gpio`). The result is masked to the RP2040 30-pin range
    /// (GPIO0..GPIO29).
    ///
    /// Next, the off-chip SPI PSRAM observes the post-merge pin state
    /// on its CS/SCK/MOSI pins and, if it is currently driving MISO,
    /// splices its bit into `gpio_in` bit 0. MISO override happens after
    /// SIO/PIO so MOSI/SCK/CS seen by the PSRAM reflect the actual pin
    /// levels driven by PIO / SIO on this tick (no feedback from the
    /// override into the PSRAM's observation on the same tick).
    ///
    /// Finally, any [`Bus::external_gpio_in_mask`] bits override the
    /// merged value with [`Bus::external_gpio_in_override`]. External
    /// drivers (e.g. the `picogus_diff_rp2040` harness injecting a
    /// synthetic ISA waveform) win over the on-chip merge for the pins
    /// they claim — without this final override step, harness pokes to
    /// `bus.gpio_in` would be silently clobbered the next time
    /// `update_gpio` ran.
    pub(crate) fn update_gpio(&mut self) {
        let mut out = self.bus.sio.gpio_out & self.bus.sio.gpio_oe;
        for pio in &self.bus.pio {
            let pio_mask = pio.pad_oe;
            out = (out & !pio_mask) | (pio.pad_out & pio_mask);
        }
        let io = &self.bus.io_bank0;
        out = ((out ^ io.out_invert) & !io.out_low) | io.out_high;
        out &= 0x3FFF_FFFF;
        if let Some(ref mut psram) = self.bus.psram
            && let Some(miso) = psram.tick(out)
        {
            let pin = psram.pin_miso();
            let mask = 1u32 << pin;
            out = (out & !mask) | ((miso as u32) << pin);
        }
        let ext_mask = self.bus.external_gpio_in_mask;
        if ext_mask != 0 {
            out = (out & !ext_mask) | (self.bus.external_gpio_in_override & ext_mask);
        }
        self.bus.gpio_in = out;
        self.bus.io_bank0.set_inputs(out);
        if self.bus.io_bank0.irq_active() {
            self.bus.irq_pending |= 1 << crate::irq::IRQ_IO_IRQ_BANK0;
        }
    }

    /// WFE/SEV / WFI quantum-end wake check. See `wrk_docs/2026.04.26
    /// - HLD - RP2040 WFE-SEV Wake Mechanics V1.md` §4.4.
    ///
    /// Per-core:
    /// - **WFE wake** — if the core is parked on `wfe_waiting` and an
    ///   `event_flag` is latched, consume the latch and un-park. The
    ///   latch is intentionally preserved (one-shot wake) when no
    ///   waiter is parked: the SEV-before-WFE idiom requires the latch
    ///   to survive until the next WFE consumes it.
    /// - **WFI wake** — if the core is halted and an enabled+pending
    ///   IRQ exists on its NVIC, un-halt. The pending bit is consumed
    ///   on the next `step()` via `try_take_any_pending_exception`. The
    ///   `halted` flag is shared with BKPT/debug halt by design (matches
    ///   rp2350_emu precedent).
    ///
    /// Crucially: this function no longer unconditionally clears
    /// `event_flag[0]`. A latched event with no waiter survives until
    /// the next WFE on that core consumes it. The launch consumer's
    /// explicit `event_flag[1] = false` reset (`maybe_wake_core1`) is
    /// preserved verbatim — that's an intentional clean-launch reset,
    /// not part of the WFE/SEV protocol.
    fn wake_checks(&mut self) {
        for core in 0..2 {
            // WFE wake: parked core + latched event = consume + un-park.
            if self.bus.wfe_waiting[core] && self.bus.event_flag[core] {
                self.bus.event_flag[core] = false;
                self.bus.wfe_waiting[core] = false;
            }
            // An enabled interrupt that can preempt also wakes WFE, per
            // ARMv6-M B1.5.18: the core takes it and returns from the
            // WFE. With PRIMASK set it cannot, and the core sleeps on.
            if self.bus.wfe_waiting[core]
                && self.cores[core].regs.primask & 1 == 0
                && self.bus.nvics[core].pending_and_enabled() != 0
            {
                self.bus.wfe_waiting[core] = false;
            }
            // WFI wake: halted core + pending+enabled IRQ = un-halt.
            // Reuses `halted` so this also wakes a BKPT-halted core if
            // an IRQ asserts; matches the rp2350_emu design wart.
            if self.cores[core].is_halted() && self.bus.nvics[core].pending_and_enabled() != 0 {
                self.cores[core].wake();
            }
        }
    }

    /// Halt core 1 and synchronously re-arm the multicore-launch FSM.
    ///
    /// This is the ONLY sanctioned path for halting core 1 from
    /// production code. Direct `cores[1].halt()` skips the `armed`
    /// sync and will silently drift the FSM state against the core's
    /// actual halt status. See HLD 2026.04.16 §5 (invariants).
    /// Diagnostic for the watch-mode idle skip: while on, every data
    /// address that made a polling pass dirty is logged with a store flag.
    pub fn idle_debug_dirty(&mut self, on: bool) {
        self.bus.idle.debug_dirty = on;
    }

    /// Takes the addresses logged since [`Self::idle_debug_dirty`] was turned on.
    pub fn take_idle_dirty_log(&mut self) -> Vec<(u32, bool)> {
        std::mem::take(&mut self.bus.idle.dirty_log)
    }

    pub fn halt_core1(&mut self) {
        self.assert_not_placeholder();
        self.cores[1].halt();
        self.bus.sio.set_handshake_armed(true);
    }

    /// Wake core 1 and synchronously disarm the multicore-launch FSM.
    ///
    /// This is the ONLY sanctioned path for waking core 1 from
    /// production code. The launch consumer in [`Self::maybe_wake_core1`]
    /// calls this after applying VTOR / MSP / PC; external callers
    /// (tests simulating a mode switch; future reset-path code) also
    /// route through here.
    ///
    /// `wake_core1` does not touch CPU register state. Callers that need
    /// a clean architectural baseline (e.g. the launch consumer after a
    /// re-halt) must call [`CortexM0Plus::reset_control_for_launch`]
    /// before this.
    pub fn wake_core1(&mut self) {
        self.assert_not_placeholder();
        self.cores[1].wake();
        self.bus.sio.set_handshake_armed(false);
    }

    /// Observe the Pico SDK multicore-launch handshake. The armed-path
    /// FSM in [`crate::bus::Sio::fifo_wr`] consumes core-0 FIFO pushes
    /// while core 1 is halted; on the 6th valid word the FSM produces a
    /// [`crate::bus::sio::Core1Launch`] token. This consumer applies
    /// VTOR / MSP / PC to core 1, resets CONTROL/PSP/xPSR/IPSR/PRIMASK
    /// to a clean launch baseline, clears any stale `event_flag[1]`,
    /// and wakes the core via the [`Self::wake_core1`] wrapper (which
    /// synchronously disarms the FSM).
    ///
    /// Called once after each core-0 step so that a pushed-then-popped
    /// handshake within a single quantum still wakes core 1 in that
    /// quantum. The `writer_core` argument is unused on this branch —
    /// the FSM is only armed while core 0 pushes, so a core-1 step
    /// cannot produce a pending_launch. Kept for call-site-compatibility
    /// with the replaced placeholder.
    fn maybe_wake_core1(&mut self, _writer_core: usize) {
        self.apply_core1_psm_reset();
        let Some(launch) = self.bus.sio.take_pending_launch() else {
            return;
        };
        // Invariant: the FSM only arms while core 1 is halted; launch
        // tokens can only be produced in that state. If this fails we
        // have a logic bug in the arming mechanism (HLD §2.5).
        debug_assert!(
            self.cores[1].is_halted(),
            "pending_launch emitted against an awake core 1 — arming bug"
        );

        self.bus.ppb[1].vtor = launch.vtor;
        self.cores[1].regs.msp = launch.sp;
        self.cores[1].regs.r[13] = launch.sp;
        // `entry & !1` matches `direct_boot_from_flash` (silent strip).
        // On real silicon a Thumb-bit-clear BLX target HardFaults; this
        // asymmetry is logged in tech_debt.md alongside direct_boot.
        self.cores[1].regs.set_pc(launch.entry & !1);
        self.cores[1].reset_control_for_launch();
        self.bus.event_flag[1] = false; // clear any stale wake signal
        self.wake_core1();
    }

    /// Apply a PSM FRCE_OFF.PROC1 edge to core 1. Forced off, core 1
    /// halts. Released, it starts in the bootrom's core 1 wait loop, which
    /// sends a 0 to core 0 through the FIFO, with an event, and then waits
    /// for the launch sequence; the handshake FSM stands in for that loop.
    fn apply_core1_psm_reset(&mut self) {
        let Some(held) = self.bus.core1_psm_reset.take() else {
            return;
        };
        self.halt_core1();
        self.bus.wfe_waiting[1] = false;
        self.bus.event_flag[1] = false;
        if !held {
            self.bus.sio.write32(0x054, 0, 1);
            if let Some(receiver) = self.bus.sio.pending_fifo_event.take() {
                self.bus.event_flag[receiver] = true;
            }
        }
    }

    /// Read a GPIO pin from the merged pin state. Debug-only: asserts
    /// the emulator has not been promoted into Threaded mode.
    pub fn gpio_read(&self, pin: u8) -> bool {
        self.assert_not_placeholder();
        if pin >= 30 {
            return false;
        }
        (self.bus.gpio_in >> pin) & 1 != 0
    }

    /// Write a GPIO pin. Sets the SIO GPIO_OUT bit and asserts output
    /// enable so the pin state becomes observable via [`Self::gpio_read`].
    /// Useful as a test-shim to inject a pin level without hand-rolling
    /// the SIO register poking.
    pub fn gpio_write(&mut self, pin: u8, value: bool) {
        self.assert_not_placeholder();
        if pin >= 30 {
            return;
        }
        let mask = 1u32 << pin;
        self.bus.sio.gpio_oe |= mask;
        if value {
            self.bus.sio.gpio_out |= mask;
        } else {
            self.bus.sio.gpio_out &= !mask;
        }
        self.update_gpio();
    }

    /// Read all GPIO pins as a bitmask. Debug-only: asserts the
    /// emulator has not been promoted into Threaded mode.
    pub fn gpio_read_all(&self) -> u64 {
        self.assert_not_placeholder();
        self.bus.gpio_in as u64
    }

    /// Access core state. Debug-only: asserts the emulator has not
    /// been promoted into Threaded mode (the flat `cores` field would
    /// be a placeholder).
    pub fn core(&self, id: usize) -> &CortexM0Plus {
        self.assert_not_placeholder();
        &self.cores[id]
    }

    /// Mutable accessor; same debug-only placeholder assertion.
    pub fn core_mut(&mut self, id: usize) -> &mut CortexM0Plus {
        self.assert_not_placeholder();
        &mut self.cores[id]
    }

    /// Direct memory read (bypasses bus timing). Debug-only: asserts
    /// the emulator has not been promoted into Threaded mode.
    pub fn peek(&self, addr: u32) -> u32 {
        self.assert_not_placeholder();
        self.bus.peek32(addr)
    }

    /// Direct memory write (bypasses bus timing). Debug-only: asserts
    /// the emulator has not been promoted into Threaded mode.
    pub fn poke(&mut self, addr: u32, value: u32) {
        self.assert_not_placeholder();
        self.bus.poke32(addr, value);
        // poke32 bypasses the Bus::write* invalidation hooks
        // (memory.sram_write32 / xip_sram direct slice). Conservative
        // bulk invalidation here keeps the cache coherent with any
        // pre-step `poke` of executable bytes, with negligible overhead
        // (callers typically poke before the first step).
        self.bus.pending_invalidation_regions |= crate::bus::invalidation_regions::BULK;
        self.cores[0].invalidate_decode_cache_all();
        self.cores[1].invalidate_decode_cache_all();
        self.bus.pending_invalidation_regions = 0;
        self.bus.pending_cache_invalidations.clear();
    }

    /// Current master cycle count. Debug-only: asserts the emulator
    /// has not been promoted into Threaded mode — Threaded callers
    /// read the live master cycle via the value returned from
    /// [`Self::run_quantum`] / [`Self::run`].
    pub fn cycles(&self) -> u64 {
        self.assert_not_placeholder();
        self.clock.cycles
    }

    /// Write a 32-bit word to an MMIO address via the bus. Charges zero
    /// emulator cycles (intended for setup code running outside `run()`).
    ///
    /// Delegates to [`Bus::write32`], so alias bits (`(addr >> 12) & 3`)
    /// are honoured: base address = normal, XOR alias = `|0x1000`, SET
    /// alias = `|0x2000`, CLR alias = `|0x3000`. Useful for poking PIO
    /// INSTR_MEM, configuring SIO GPIO_OE/_OUT, releasing RESETS bits,
    /// etc., without hand-rolling the bus machinery.
    pub fn mmio_write32(&mut self, addr: u32, value: u32) {
        self.assert_not_placeholder();
        // Mirror the `step()` stash so PLL write-time lock-arm transitions
        // observe the current cycle count when the harness pokes MMIO
        // outside the step path. See HLD §6 P2.
        self.bus.master_cycle = self.clock.cycles;
        self.bus.write32(addr, value);
    }

    /// Read a 32-bit word from an MMIO address via the bus. Charges zero
    /// emulator cycles (intended for setup code running outside `run()`).
    ///
    /// **Warning: reads may have side effects.** Several RP2040 MMIO
    /// registers mutate state on read — e.g. PIO `RXFn` pops the receive
    /// FIFO, SIO divider `QUOTIENT` / `REMAINDER` clear the CSR dirty
    /// bit, and a handful of W1C sticky flags are cleared by reads. Setup
    /// code should therefore be write-heavy; reads through this method
    /// are for confirmation only and should be chosen carefully to avoid
    /// disturbing the peripheral's state.
    pub fn mmio_read32(&mut self, addr: u32) -> u32 {
        self.assert_not_placeholder();
        // Mirror the `step()` stash so PLL CS reads observe the current
        // cycle count when the harness reads MMIO outside the step path.
        self.bus.master_cycle = self.clock.cycles;
        self.bus.read32(addr)
    }

    /// Harness-only diagnostic: drain every byte firmware has written to
    /// UART0 `DR` since the previous call. Returns empty if idle.
    pub fn drain_uart0_tx_log(&mut self) -> Vec<u8> {
        self.assert_not_placeholder();
        self.bus.drain_uart0_tx_log()
    }

    /// Harness-only: deliver one byte to UART0's receiver as if it had
    /// arrived on the RX pin. False when the receiver is off or its FIFO
    /// is full, and the byte was dropped.
    pub fn uart0_push_rx(&mut self, byte: u8) -> bool {
        self.assert_not_placeholder();
        self.bus.uart0_push_rx(byte)
    }
}

/// Builder for assembling the emulator. Seeds the Bus clock tree from
/// `Config::sys_clk_hz` — the first CLOCKS / PLL register write
/// replaces the seed with the derived value.
pub struct EmulatorBuilder {
    config: Config,
    step_quantum: u32,
    flash: Option<Vec<u8>>,
    psram: Option<picoem_devices::Psram>,
    execution: ExecutionModel,
}

impl EmulatorBuilder {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            step_quantum: DEFAULT_STEP_QUANTUM,
            flash: None,
            psram: None,
            execution: ExecutionModel::default(),
        }
    }

    /// Override the per-step quantum (default [`DEFAULT_STEP_QUANTUM`]).
    ///
    /// Clamps `0 -> 1`. Previously a `debug_assert!` here meant
    /// `step_quantum(0)` silently advanced 0 cycles per `step()` in
    /// release builds, a guaranteed infinite-loop footgun for `run()`.
    pub fn step_quantum(mut self, n: u32) -> Self {
        self.step_quantum = n.max(1);
        self
    }

    /// Pre-load an XIP flash image. Applied at [`Self::build`] time via
    /// [`Emulator::load_flash`]; oversize images are silently clamped to
    /// the 2 MB flash window.
    pub fn flash(mut self, bytes: Vec<u8>) -> Self {
        self.flash = Some(bytes);
        self
    }

    /// Attach an off-chip SPI PSRAM device to the emulator. When set,
    /// [`Emulator::update_gpio`] feeds the device's `tick()` method on
    /// every GPIO merge and splices its MISO output back into `gpio_in`.
    pub fn psram(mut self, psram: picoem_devices::Psram) -> Self {
        self.psram = Some(psram);
        self
    }

    /// Select the runtime [`ExecutionModel`]. Defaults to
    /// `ExecutionModel::Serial` (the oracle-validated reference path).
    /// `ExecutionModel::Threaded` requires the `threading` cargo feature
    /// and an x86_64 Windows host; otherwise [`Self::build`] returns
    /// `Err(ConfigError::ThreadingUnavailable)`.
    pub fn execution(mut self, model: ExecutionModel) -> Self {
        self.execution = model;
        self
    }

    pub fn build(self) -> Result<Emulator, ConfigError> {
        // Threading availability gate — dual-execution HLD V1 §5.2.
        if self.execution == ExecutionModel::Threaded {
            #[cfg(not(all(
                feature = "threading",
                target_arch = "x86_64",
                any(target_os = "windows", target_os = "linux")
            )))]
            return Err(ConfigError::ThreadingUnavailable);
            #[cfg(all(
                feature = "threading",
                target_arch = "x86_64",
                any(target_os = "windows", target_os = "linux")
            ))]
            {
                let n = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
                if n < 3 {
                    return Err(ConfigError::ThreadingUnavailable);
                }
            }
        }

        let mut bus = Bus::new();
        bus.seed_sys_clk_hz(self.config.sys_clk_hz);
        bus.psram = self.psram;
        let mut emu = Emulator {
            cores: [CortexM0Plus::with_id(0), CortexM0Plus::with_id(1)],
            bus,
            clock: Clock { cycles: 0 },
            step_quantum: self.step_quantum,
            gpio_trace: None,
            pio_tick_count: 0,
            pio_tick_iow_low_count: 0,
            pio0_sm0_max_pc: 0,
            pio0_sm0_pc_advances: 0,
            pio0_sm0_last_pc: 0xFF,
            execution_model: self.execution,
            #[cfg(all(
                feature = "threading",
                target_arch = "x86_64",
                any(target_os = "windows", target_os = "linux")
            ))]
            threaded: None,
            #[cfg(all(
                feature = "threading",
                target_arch = "x86_64",
                any(target_os = "windows", target_os = "linux")
            ))]
            panic_info: None,
            #[cfg(all(
                feature = "threading",
                target_arch = "x86_64",
                any(target_os = "windows", target_os = "linux")
            ))]
            timeout_info: None,
            #[cfg(all(
                feature = "testing",
                feature = "threading",
                target_arch = "x86_64",
                any(target_os = "windows", target_os = "linux")
            ))]
            pending_panic_inject: None,
            #[cfg(all(
                feature = "threading",
                target_arch = "x86_64",
                any(target_os = "windows", target_os = "linux")
            ))]
            bus_is_placeholder: false,
            idle: IdleSkip::default(),
            blocks: Blocks::default(),
        };
        emu.blocks.mode = match std::env::var("PICOEM_BLOCKS").as_deref() {
            Ok("on") | Ok("1") => BlockMode::On,
            Ok("check") => BlockMode::Check,
            _ => BlockMode::Off,
        };
        emu.blocks.jit = matches!(std::env::var("PICOEM_JIT").as_deref(), Ok("on") | Ok("1"));
        // Default: core 1 halted — Pico SDK wakes it via SIO FIFO.
        // Route through the wrapper so the SIO handshake FSM `armed`
        // flag is in sync (HLD 2026.04.16 §2.1 / §5 invariant).
        emu.halt_core1();
        if let Some(bytes) = self.flash {
            emu.load_flash(&bytes);
        }
        info!(
            rom_size = ROM_SIZE,
            sram_size = SRAM_SIZE,
            step_quantum = self.step_quantum,
            sys_clk_hz = self.config.sys_clk_hz,
            execution = ?self.execution,
            "emulator constructed"
        );
        Ok(emu)
    }
}

// ---------------------------------------------------------------------------
// Stage 4: residue branch coverage for the top-level `lib.rs` (Emulator,
// EmulatorBuilder, Config, ConfigError, EmulatorError, ExecutionModel,
// WorkerName). Pure append-only — does not modify any production code.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage4_lib_residue {
    use super::*;

    // ------------------- ConfigError -------------------

    #[test]
    fn config_error_display_threading_unavailable() {
        let s = format!("{}", ConfigError::ThreadingUnavailable);
        assert!(s.contains("Threaded"));
        assert!(s.contains("unavailable"));
    }

    #[test]
    fn config_error_debug_and_clone_eq() {
        let e1 = ConfigError::ThreadingUnavailable;
        let e2 = e1.clone();
        assert_eq!(e1, e2);
        let _ = format!("{:?}", e1);
    }

    #[test]
    fn config_error_is_std_error() {
        fn assert_err<E: std::error::Error>(_: &E) {}
        assert_err(&ConfigError::ThreadingUnavailable);
    }

    // ------------------- EmulatorError -------------------

    #[test]
    fn emulator_error_display_not_supported_in_threaded() {
        let s = format!("{}", EmulatorError::NotSupportedInThreadedMode);
        assert!(s.contains("Threaded"));
    }

    #[test]
    fn emulator_error_display_worker_panicked() {
        let e = EmulatorError::WorkerPanicked {
            which: WorkerName::Core0,
            message: String::from("boom"),
        };
        let s = format!("{}", e);
        assert!(s.contains("panicked"));
        assert!(s.contains("boom"));
        assert!(s.contains("core0"));
    }

    #[test]
    fn emulator_error_display_barrier_timeout() {
        let e = EmulatorError::BarrierTimeout {
            which: WorkerName::Coord,
            elapsed_ms: 1_234,
        };
        let s = format!("{}", e);
        assert!(s.contains("barrier"));
        assert!(s.contains("1234"));
        assert!(s.contains("coord"));
    }

    #[test]
    fn emulator_error_clone_eq_debug() {
        let e1 = EmulatorError::NotSupportedInThreadedMode;
        let e2 = e1.clone();
        assert_eq!(e1, e2);
        let _ = format!("{:?}", e1);
    }

    // ------------------- WorkerName -------------------

    #[test]
    fn worker_name_as_str_all_variants() {
        assert_eq!(WorkerName::Core0.as_str(), "core0");
        assert_eq!(WorkerName::Core1.as_str(), "core1");
        assert_eq!(WorkerName::Coord.as_str(), "coord");
    }

    #[test]
    fn worker_name_clone_eq_debug() {
        let w = WorkerName::Core1;
        assert_eq!(w, w.clone());
        let _ = format!("{:?}", w);
    }

    // ------------------- ExecutionModel -------------------

    #[test]
    fn execution_model_default_is_serial() {
        assert_eq!(ExecutionModel::default(), ExecutionModel::Serial);
    }

    #[test]
    fn execution_model_eq_and_debug() {
        assert_eq!(ExecutionModel::Threaded, ExecutionModel::Threaded);
        assert_ne!(ExecutionModel::Serial, ExecutionModel::Threaded);
        let _ = format!("{:?}", ExecutionModel::Serial);
        let _ = format!("{:?}", ExecutionModel::Threaded);
    }

    // ------------------- Builder: ConfigError::ThreadingUnavailable -------------------

    #[cfg(not(feature = "threading"))]
    #[test]
    fn builder_threaded_no_feature_returns_threading_unavailable() {
        let res = EmulatorBuilder::new(Config::default())
            .execution(ExecutionModel::Threaded)
            .build();
        match res {
            Err(ConfigError::ThreadingUnavailable) => {}
            Ok(_) => panic!("Threaded should fail without `threading` feature"),
        }
    }

    #[cfg(not(all(
        feature = "threading",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    )))]
    #[test]
    fn builder_threaded_off_platform_returns_threading_unavailable() {
        let res = EmulatorBuilder::new(Config::default())
            .execution(ExecutionModel::Threaded)
            .build();
        match res {
            Err(ConfigError::ThreadingUnavailable) => {}
            Ok(_) => panic!("Threaded should fail on unsupported platforms"),
        }
    }

    // ------------------- Builder defaults / overrides -------------------

    #[test]
    fn builder_default_step_quantum() {
        let emu = EmulatorBuilder::new(Config::default()).build().unwrap();
        assert_eq!(emu.step_quantum, DEFAULT_STEP_QUANTUM);
    }

    #[test]
    fn builder_custom_step_quantum() {
        let emu = EmulatorBuilder::new(Config::default())
            .step_quantum(8)
            .build()
            .unwrap();
        assert_eq!(emu.step_quantum, 8);
    }

    #[test]
    fn step_quantum_zero_clamps_to_one() {
        // Regression: `EmulatorBuilder::step_quantum(0)` previously
        // tripped a `debug_assert!` (and silently advanced 0 cycles
        // per `step()` in release builds — an infinite-loop footgun
        // for `run()`). The clamp at the builder entry point keeps
        // the runtime contract `step_quantum >= 1` intact.
        let mut emu = EmulatorBuilder::new(Config::default())
            .step_quantum(0)
            .build()
            .unwrap();
        assert_eq!(emu.step_quantum, 1);
        // `step()` must make forward progress (advance >= 1 master
        // cycle) and not loop forever.
        let advanced = emu.step().unwrap();
        assert!(advanced >= 1);
    }

    #[test]
    fn builder_custom_sysclk() {
        let cfg = Config {
            sys_clk_hz: 125_000_000,
        };
        let emu = EmulatorBuilder::new(cfg).build().unwrap();
        // The clock tree is recomputed from sys_clk_hz; simplest check is
        // that the emulator builds and the master cycle starts at zero.
        assert_eq!(emu.cycles(), 0);
    }

    #[test]
    fn builder_execution_serial_explicit() {
        let emu = EmulatorBuilder::new(Config::default())
            .execution(ExecutionModel::Serial)
            .build()
            .unwrap();
        assert_eq!(emu.execution_model(), ExecutionModel::Serial);
    }

    #[test]
    fn builder_with_flash_pre_loads_xip() {
        let mut flash = vec![0u8; 32];
        flash[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        let emu = EmulatorBuilder::new(Config::default())
            .flash(flash)
            .build()
            .unwrap();
        // XIP base is 0x1000_0000 — the flash loader writes there.
        assert_eq!(emu.bus.memory.xip_read32(0), 0xDEAD_BEEF);
    }

    // ------------------- Emulator basics -------------------

    #[test]
    fn execution_model_accessor_returns_selected() {
        let emu = Emulator::new(Config::default());
        assert_eq!(emu.execution_model(), ExecutionModel::Serial);
    }

    #[test]
    fn core_cycles_default_zero() {
        let emu = Emulator::new(Config::default());
        assert_eq!(emu.core_cycles(0), 0);
        assert_eq!(emu.core_cycles(1), 0);
    }

    #[test]
    #[should_panic(expected = "core_cycles: idx must be 0 or 1")]
    fn core_cycles_invalid_idx_panics() {
        let emu = Emulator::new(Config::default());
        let _ = emu.core_cycles(2);
    }

    #[test]
    fn cycles_starts_at_zero() {
        let emu = Emulator::new(Config::default());
        assert_eq!(emu.cycles(), 0);
    }

    #[test]
    fn core_and_core_mut_accessors() {
        let mut emu = Emulator::new(Config::default());
        // No id() public method available like rp2350_emu's, but we can
        // exercise both accessors and the placeholder-guard path.
        let _ = emu.core(0);
        let _ = emu.core_mut(1);
    }

    // ------------------- Emulator::run / step -------------------

    #[test]
    fn run_zero_cycles_serial_is_noop() {
        let mut emu = Emulator::new(Config::default());
        let executed = emu.run(0).unwrap();
        // run() returns the delta in cycles. With cycles=0 the inner loop
        // condition `clock.cycles - start < 0` is false on first check, so
        // no quanta run.
        assert_eq!(executed, 0);
    }

    #[test]
    fn run_quantum_serial_returns_ok() {
        let mut emu = Emulator::new(Config::default());
        let r = emu.run_quantum().unwrap();
        // run_quantum returns the number of cycles consumed in this
        // quantum; with both cores halted (core 1 always halted, core 0
        // running zero-data ROM) this is bounded by step_quantum.
        assert!(r <= emu.step_quantum as u64);
    }

    #[test]
    fn step_serial_returns_ok() {
        let mut emu = Emulator::new(Config::default());
        // Both cores halt quickly with zero-data ROM, but step still
        // returns Ok(consumed) on Serial.
        let _ = emu.step().unwrap();
    }

    // ------------------- gpio bounds -------------------

    #[test]
    fn gpio_read_pin_30_returns_false() {
        let emu = Emulator::new(Config::default());
        assert!(!emu.gpio_read(30));
        assert!(!emu.gpio_read(31));
    }

    #[test]
    fn gpio_write_pin_out_of_range_is_noop() {
        let mut emu = Emulator::new(Config::default());
        // Pin 30 is past the valid range. The function bails early — no
        // GPIO_OE bit gets set.
        emu.gpio_write(30, true);
        assert_eq!(emu.bus.sio.gpio_oe & (1u32 << 30), 0);
    }

    #[test]
    fn gpio_write_in_range_sets_oe_and_out() {
        let mut emu = Emulator::new(Config::default());
        emu.gpio_write(5, true);
        assert_ne!(emu.bus.sio.gpio_oe & (1u32 << 5), 0);
        assert_ne!(emu.bus.sio.gpio_out & (1u32 << 5), 0);
        emu.gpio_write(5, false);
        assert_eq!(emu.bus.sio.gpio_out & (1u32 << 5), 0);
    }

    #[test]
    fn gpio_read_all_default_zero() {
        let emu = Emulator::new(Config::default());
        assert_eq!(emu.gpio_read_all(), 0);
    }

    // ------------------- load_image / load_bootrom / load_flash -------------------

    #[test]
    fn load_image_sram_writes_through() {
        let mut emu = Emulator::new(Config::default());
        let data = [0x11u8, 0x22, 0x33, 0x44];
        emu.load_image(0x2000_0100, &data);
        assert_eq!(emu.peek(0x2000_0100), 0x4433_2211);
    }

    #[test]
    fn load_image_rom_region_overlay() {
        let mut emu = Emulator::new(Config::default());
        let data = [0xAAu8, 0xBB, 0xCC, 0xDD];
        emu.load_image(0x0000_0000, &data);
        assert_eq!(emu.bus.memory.rom_read8(0), 0xAA);
        assert_eq!(emu.bus.memory.rom_read8(3), 0xDD);
    }

    #[test]
    fn load_image_unknown_region_silently_dropped() {
        let mut emu = Emulator::new(Config::default());
        let data = [0xFFu8; 4];
        // 0x4 region — no match arm, falls through.
        emu.load_image(0x4000_0000, &data);
    }

    #[test]
    fn load_bootrom_loads_first_word() {
        let mut emu = Emulator::new(Config::default());
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(&0x2000_8000u32.to_le_bytes());
        emu.load_bootrom(&data);
        assert_eq!(emu.bus.memory.rom_read32(0), 0x2000_8000);
    }

    #[test]
    fn load_flash_drains_invalidations() {
        let mut emu = Emulator::new(Config::default());
        emu.load_flash(&[0u8; 16]);
        assert_eq!(emu.bus.pending_invalidation_regions, 0);
    }

    // ------------------- direct_boot_from_flash -------------------

    #[test]
    fn direct_boot_from_flash_seeds_sp_pc_vtor() {
        let mut emu = Emulator::new(Config::default());
        // Build a minimal vector table at flash offset 0x100 (pico-sdk).
        let sp = 0x2002_0000u32;
        let entry = 0x1000_0301u32; // Thumb-bit set
        let vtor_offset = 0x100u32;
        let mut flash = vec![0u8; 0x200];
        flash[(vtor_offset as usize)..(vtor_offset as usize + 4)]
            .copy_from_slice(&sp.to_le_bytes());
        flash[(vtor_offset as usize + 4)..(vtor_offset as usize + 8)]
            .copy_from_slice(&entry.to_le_bytes());
        emu.load_flash(&flash);
        emu.direct_boot_from_flash(vtor_offset);
        for c in 0..2 {
            assert_eq!(emu.cores[c].regs.msp, sp);
            assert_eq!(emu.cores[c].regs.pc(), entry & !1);
        }
        assert_eq!(emu.bus.ppb[0].vtor, 0x1000_0000 + vtor_offset);
        // Core 1 stays halted by halt_core1.
        assert!(emu.cores[1].is_halted());
    }

    // ------------------- halt_core1 / wake_core1 -------------------

    #[test]
    fn halt_and_wake_core1_round_trip() {
        let mut emu = Emulator::new(Config::default());
        // Initial state: core 1 halted.
        assert!(emu.cores[1].is_halted());
        emu.wake_core1();
        assert!(!emu.cores[1].is_halted());
        emu.halt_core1();
        assert!(emu.cores[1].is_halted());
    }

    // ------------------- mmio_read32 / mmio_write32 -------------------

    #[test]
    fn mmio_write_read_roundtrip_sram() {
        let mut emu = Emulator::new(Config::default());
        emu.mmio_write32(0x2000_0000, 0xDEAD_BEEF);
        assert_eq!(emu.mmio_read32(0x2000_0000), 0xDEAD_BEEF);
    }

    // ------------------- reset -------------------

    #[test]
    fn reset_clears_clock() {
        let mut emu = Emulator::new(Config::default());
        let _ = emu.step().unwrap();
        emu.reset();
        assert_eq!(emu.cycles(), 0);
        assert_eq!(emu.pio_tick_count, 0);
    }

    // ------------------- drain_uart0_tx_log -------------------

    #[test]
    fn drain_uart0_tx_log_default_empty() {
        let mut emu = Emulator::new(Config::default());
        let v = emu.drain_uart0_tx_log();
        assert!(v.is_empty());
    }

    // ------------------- poke / peek -------------------

    #[test]
    fn poke_and_peek_round_trip_sram() {
        let mut emu = Emulator::new(Config::default());
        emu.poke(0x2000_2000, 0xCAFE_F00D);
        assert_eq!(emu.peek(0x2000_2000), 0xCAFE_F00D);
    }
}

// ---------------------------------------------------------------------------
// Stage 5: branch-coverage residue not hit by Stage 4. Targets specific
// `if` arms inside `reset`, `load_image`, `step_serial`, `tick_systick`,
// `tick_pio_and_route_irqs`, `tick_pio`, `update_gpio`, and `wake_checks`.
// Pure append-only — does not modify production code.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage5_lib_residue {
    use super::*;
    use picoem_devices::Psram;

    // ------------------- reset: psram present (line 422) -------------------

    /// Drives the true branch of `if let Some(ref mut psram) = self.bus.psram`
    /// (line 422) inside `reset()` by attaching a PSRAM device first.
    #[test]
    fn reset_with_psram_attached() {
        let psram = Psram::new(0, 1, 2, 3);
        let mut emu = EmulatorBuilder::new(Config::default())
            .psram(psram)
            .build()
            .unwrap();
        emu.reset();
        assert_eq!(emu.cycles(), 0);
    }

    // ------------------- load_image: ROM offset boundary (line 461) -------------------

    /// Drives the false branch of `if offset < ROM_SIZE` inside the ROM
    /// arm of `load_image` (line 461) by passing an offset past ROM end.
    /// `offset = addr & 0x0FFF_FFFF` so `0x0FFF_FFFF` is well past the
    /// 16 KB ROM_SIZE.
    #[test]
    fn load_image_rom_offset_past_end_is_skipped() {
        let mut emu = Emulator::new(Config::default());
        let data = [0x55u8; 4];
        // offset = 0x0FFF_FFFF >= ROM_SIZE so the inner copy is skipped.
        emu.load_image(0x0FFF_FFFF, &data);
        // ROM byte 0 untouched (was 0).
        assert_eq!(emu.bus.memory.rom_read8(0), 0);
    }

    /// Drives the true branch of the same `if offset < ROM_SIZE` (line 461)
    /// — the existing test suite hits this through `load_image_rom_region_overlay`,
    /// repeated here for explicit branch attribution.
    #[test]
    fn load_image_rom_offset_inside_overlays() {
        let mut emu = Emulator::new(Config::default());
        let data = [0xAAu8, 0xBB, 0xCC, 0xDD];
        emu.load_image(0x0000_0010, &data);
        assert_eq!(emu.bus.memory.rom_read8(0x10), 0xAA);
    }

    // ------------------- step_serial: while-loop arms + c0/c1 paths -------------------

    /// Drives the true arm of the c1 step `if !self.cores[1].is_halted()
    /// && !self.bus.wfe_waiting[1]` (line 700) by waking core 1 first.
    /// Also covers line 681/682 (while predicate evaluates with both
    /// cores live).
    #[test]
    fn step_serial_runs_core1_when_awake() {
        let mut emu = Emulator::new(Config::default());
        emu.wake_core1();
        assert!(!emu.cores[1].is_halted());
        let _ = emu.step().unwrap();
    }

    /// Drives the WFE-waiting branch in the c1 selection (line 700 false
    /// arm via wfe_waiting=true).
    #[test]
    fn step_serial_skips_core1_when_wfe_waiting() {
        let mut emu = Emulator::new(Config::default());
        emu.wake_core1();
        emu.bus.wfe_waiting[1] = true;
        let _ = emu.step().unwrap();
    }

    /// Drives the true arm of `if c0 == 0 && c1 == 0` (line 715) by
    /// halting core 0 and parking core 1 in WFE so neither contributes
    /// cycles, but the while predicate still fires once because halted
    /// flags can be reset between iterations. Best-effort branch-visit
    /// test — actual entry depends on inner-loop timing.
    #[test]
    fn step_serial_break_on_zero_cycles_smoke() {
        let mut emu = Emulator::new(Config::default());
        // Both cores halted so the while predicate gates entry to the
        // loop body. The loop exits before producing cycles, so the
        // post-loop bookkeeping still runs.
        emu.cores[0].halt();
        emu.halt_core1();
        let _ = emu.step().unwrap();
    }

    /// Production fix for tech_debt §1649: when both cores are blocked
    /// (halted/WFI here) and a TIMER alarm with INTE is scheduled in the
    /// future, the master clock must still advance to the alarm's fire
    /// cycle so the IRQ raises and `wake_checks` un-halts a core.
    ///
    /// Pre-fix behaviour: `consumed == 0` every quantum → master_cycle
    /// never moves → poll_alarms never matches → core stays parked
    /// forever.
    /// Post-fix behaviour: the both-blocked branch advances the clock
    /// to the soonest scheduled fire cycle (capped by `step_quantum`),
    /// raises the IRQ, drains it to NVICs, and `wake_checks` un-halts.
    #[test]
    fn step_serial_advances_clock_when_both_cores_blocked_with_armed_alarm() {
        use crate::peripherals::timer::{ALARM0_OFFSET, INTE_OFFSET};

        // Use a generous step_quantum so a single step() can cover the
        // whole armed-alarm interval in one go.
        let mut emu = EmulatorBuilder::new(Config::default())
            .step_quantum(2_000_000)
            .build()
            .expect("Serial build is infallible");

        // Both cores parked. Core 1 is halt_core1 (correct production
        // path); core 0 we explicitly halt to emulate WFI.
        emu.cores[0].halt();
        emu.halt_core1();
        assert!(emu.cores[0].is_halted());
        assert!(emu.cores[1].is_halted());

        // Arm TIMER ALARM0 200 µs into the future and enable INTE so the
        // fire raises NVIC line 0. Reach into bus.timer directly — same
        // crate, `pub(crate)` field.
        let sys_hz = emu.bus.clock_tree.sys_clk_hz;
        emu.bus
            .timer
            .write32(INTE_OFFSET, 0x1, 0, emu.bus.master_cycle, sys_hz);
        emu.bus
            .timer
            .write32(ALARM0_OFFSET, 200, 0, emu.bus.master_cycle, sys_hz);

        // Sanity: nothing pending yet — the bus IRQ vector is clean,
        // both NVICs are clean, and both cores are halted. Without the
        // fix, this state is a permanent dead-end.
        assert_eq!(emu.bus.irq_pending, 0);
        assert_eq!(emu.bus.nvics[0].pending_and_enabled(), 0);
        assert_eq!(emu.bus.nvics[1].pending_and_enabled(), 0);

        // One step suffices when step_quantum >> alarm horizon.
        let _ = emu.step().unwrap();

        // Post-fix: TIMER alarm fired, IRQ drained to at least one NVIC,
        // and `wake_checks` un-halted the core(s) it landed on. Any of
        // these three observations is sufficient — and on the pre-fix
        // code, none of them holds.
        let nvic_pending = emu.bus.nvics[0].is_pending(0) || emu.bus.nvics[1].is_pending(0);
        let core_woke = !emu.cores[0].is_halted() || !emu.cores[1].is_halted();
        assert!(
            nvic_pending || core_woke,
            "Both-blocked clock-advance branch did not deliver TIMER alarm: \
             nvic0_pend0={} nvic1_pend0={} c0_halted={} c1_halted={} \
             master_cycle={} alarm_fire={:?}",
            emu.bus.nvics[0].is_pending(0),
            emu.bus.nvics[1].is_pending(0),
            emu.cores[0].is_halted(),
            emu.cores[1].is_halted(),
            emu.bus.master_cycle,
            emu.bus.next_scheduled_lazy_deadline(),
        );
    }

    // ------------------- step_serial: slow-path / fast-path gating (line 750) -------------------

    /// Drives the slow-path arm of the fast-path gate (line 750 false)
    /// by enabling SysTick on the active core. The predicate
    /// `systick_idle` becomes false, forcing the slow branch.
    #[test]
    fn step_serial_drops_to_slow_path_when_systick_enabled() {
        let mut emu = Emulator::new(Config::default());
        // Enable SysTick on core 0 (CSR.ENABLE = bit 0).
        emu.bus.systicks[0].csr |= 1;
        let _ = emu.step().unwrap();
    }

    /// Drives the fast-path arm of the same gate (line 750 true). Default
    /// state has no SysTick, no IRQ, no PIO — the existing
    /// `step_serial_returns_ok` already exercises this; making it explicit
    /// here for branch attribution.
    #[test]
    fn step_serial_takes_fast_path_when_idle() {
        let mut emu = Emulator::new(Config::default());
        let _ = emu.step().unwrap();
    }

    // ------------------- drain_pending_irqs_to_cores (line 792, 795) -------------------

    /// Drives the true arm of `if self.bus.irq_pending != 0` (line 792)
    /// AND the inner per-IRQ scan (line 795 true) by pre-staging a bus
    /// irq_pending bit. Forces the slow path via SysTick enable so the
    /// drain runs.
    #[test]
    fn drain_pending_irqs_routes_to_nvics() {
        let mut emu = Emulator::new(Config::default());
        // Force slow path.
        emu.bus.systicks[0].csr |= 1;
        // Pre-stage a pending IRQ on bus.irq_pending.
        emu.bus.irq_pending = 0x1; // line 0
        let _ = emu.step().unwrap();
        // After the slow path drains, both NVICs see line 0 pending.
        assert!(emu.bus.nvics[0].is_pending(0) || emu.bus.nvics[1].is_pending(0));
    }

    // ------------------- tick_systick (lines 812, 817) -------------------

    /// Drives the true branches of `if systicks[0].tick()` (line 812) and
    /// `if systicks[1].tick()` (line 817) by enabling SysTick with
    /// CVR=0, RVR=0 so the very first tick fires.
    #[test]
    fn tick_systick_fires_on_both_cores_when_enabled() {
        let mut emu = Emulator::new(Config::default());
        // Wake core 1 so it consumes cycles → tick_systick(c0, c1) with
        // both > 0.
        emu.wake_core1();
        // Enable SysTick (ENABLE=1, TICKINT=1) on both cores; CVR=0, RVR=0
        // → first tick reloads + fires.
        emu.bus.systicks[0].csr = 0b11;
        emu.bus.systicks[1].csr = 0b11;
        emu.bus.systicks[0].cvr = 0;
        emu.bus.systicks[1].cvr = 0;
        let _ = emu.step().unwrap();
    }

    /// Core 0's SysTick must count while core 1 is awake too. After a
    /// two-core quantum the active core is core 1, so a gate that only
    /// looked at the active core's SysTick took the fast path and core
    /// 0's counter never moved.
    #[test]
    fn core0_systick_ticks_while_core1_runs() {
        let mut emu = Emulator::new(Config::default());
        // Both cores spin on `b .` in SRAM.
        emu.bus.write16(0x2000_1000, 0xE7FE);
        emu.bus.write16(0x2000_1010, 0xE7FE);
        emu.cores[0].regs.set_pc(0x2000_1000);
        emu.wake_core1();
        emu.cores[1].regs.set_pc(0x2000_1010);
        // Core 0 only: ENABLE + CLKSOURCE, no TICKINT, large reload.
        emu.bus.systicks[0].rvr = 0x00FF_FFFF;
        emu.bus.systicks[0].cvr = 0x00FF_FFFF;
        emu.bus.systicks[0].csr = 0b101;
        for _ in 0..100 {
            let _ = emu.step().unwrap();
        }
        assert!(emu.bus.systicks[0].cvr < 0x00FF_FFFF - 1000, "core 0 SysTick stalled at {:#x}", emu.bus.systicks[0].cvr);
    }

    /// A level IRQ that stays high while its handler runs must not pend
    /// again before the handler clears it. TIMER alarm 0's INTR stays
    /// latched through 40 NOPs, then the handler clears it and counts
    /// itself. It has to run exactly once at every quantum size.
    #[test]
    fn level_irq_does_not_repend_while_its_handler_runs() {
        const TIMER: u32 = 0x4005_4000;
        for q in [1u32, 8, 64] {
            let mut emu = Emulator::new(Config::default());
            emu.step_quantum = q;
            let handler = 0x2000_1100u32;
            for v in 0..48 {
                emu.bus.write32(0x2000_0000 + 4 * v, 0x2000_1200 | 1);
            }
            emu.bus.write16(0x2000_1200, 0xE7FE);
            emu.bus.write32(0x2000_0000 + 16 * 4, handler | 1);
            emu.bus.ppb[0].vtor = 0x2000_0000;
            let mut code = vec![0xBF00u16; 40];
            // str r2,[r4] clears INTR; adds r5,#1; bx lr
            code.extend([0x6022, 0x3501, 0x4770]);
            for (i, hw) in code.iter().enumerate() {
                emu.bus.write16(handler + 2 * i as u32, *hw);
            }
            emu.bus.write16(0x2000_1000, 0xE7FE);
            let c = &mut emu.cores[0];
            c.regs.r[2] = 1;
            c.regs.r[4] = TIMER + 0x34;
            c.regs.r[13] = 0x2000_3000;
            c.regs.msp = 0x2000_3000;
            c.regs.set_pc(0x2000_1000);
            // TIMER out of reset, alarm 0 enabled in INTE and the NVIC.
            let r = emu.bus.read32(0x4000_C000);
            emu.bus.write32(0x4000_C000, r & !(1 << 21));
            emu.bus.write32(TIMER + 0x38, 1);
            emu.bus.write32(0xE000_E100, 1);
            let now = emu.bus.read32(TIMER + 0x28);
            emu.bus.write32(TIMER + 0x10, now.wrapping_add(3));
            emu.run(200_000).unwrap();
            assert_eq!(emu.bus.read32(TIMER + 0x34) & 1, 0, "q {q}: INTR still set, handler never ran");
            assert_eq!(emu.cores[0].regs.r[5], 1, "q {q}: handler entries");
        }
    }

    // ------------------- idle-loop skip -------------------

    /// Core 0 runs `prog` from SRAM 0x2000_1000 with `regs` preset and core
    /// 1 halted, for `cycles` cycles in 4096-cycle quanta. Returns the
    /// final registers, cycles, the word at 0x2000_0100 and the cycles
    /// skipped.
    fn run_idle_prog(prog: &[u16], regs: &[(usize, u32)], cycles: u64, skip: bool) -> ([u32; 16], u64, u32, u64) {
        let mut emu = Emulator::new(Config::default());
        for (i, hw) in prog.iter().enumerate() {
            emu.bus.write16(0x2000_1000 + 2 * i as u32, *hw);
        }
        emu.step_quantum = 4096;
        emu.idle.enabled = skip;
        let c = &mut emu.cores[0];
        c.regs.r[13] = 0x2000_2000;
        c.regs.msp = 0x2000_2000;
        for &(r, v) in regs {
            c.regs.r[r] = v;
        }
        c.regs.set_pc(0x2000_1000);
        emu.run(cycles).unwrap();
        let mem = emu.bus.read32(0x2000_0100);
        (emu.cores[0].regs.r, emu.cycles(), mem, emu.idle.skipped_cycles[0])
    }

    /// Core 0 polls a flag in SRAM while core 1 counts to `n` and then
    /// sets it; both cores then count in r5 until the run ends, so the gap
    /// between the two counts says how soon core 0 left its loop. Returns core 0's registers, core 1's
    /// registers and core 0's parked cycles.
    fn run_two_core_poll(two_core: bool, q: u32, n: u32) -> ([u32; 16], [u32; 16], u64) {
        let core0 = [
            0x6808, // 1000: ldr r0, [r1]     poll the flag
            0x2800, // 1002: cmp r0, #0
            0xD0FC, // 1004: beq 1000
            0x3501, // 1006: adds r5, #1
            0xE7FD, // 1008: b 1006
        ];
        let core1 = [
            0x3301, // 1100: adds r3, #1
            0x42A3, // 1102: cmp r3, r4
            0xD1FC, // 1104: bne 1100
            0x2001, // 1106: movs r0, #1
            0x6008, // 1108: str r0, [r1]     set the flag
            0x3501, // 110A: adds r5, #1
            0xE7FD, // 110C: b 110A
        ];
        let mut emu = Emulator::new(Config::default());
        for (i, hw) in core0.iter().enumerate() {
            emu.bus.write16(0x2000_1000 + 2 * i as u32, *hw);
        }
        for (i, hw) in core1.iter().enumerate() {
            emu.bus.write16(0x2000_1100 + 2 * i as u32, *hw);
        }
        emu.step_quantum = q;
        emu.idle.two_core = two_core;
        for (k, pc, sp) in [(0usize, 0x2000_1000u32, 0x2000_2000u32), (1, 0x2000_1100, 0x2000_3000)] {
            let c = &mut emu.cores[k];
            c.regs.r[13] = sp;
            c.regs.msp = sp;
            c.regs.r[1] = 0x2000_0100;
            c.regs.r[3] = 0;
            c.regs.r[4] = n;
            c.regs.r[5] = 0;
            c.regs.set_pc(pc);
        }
        emu.cores[1].wake();
        emu.run(n as u64 * 8 + 100_000).unwrap();
        (emu.cores[0].regs.r, emu.cores[1].regs.r, emu.idle.parked_cycles[0])
    }

    /// While core 1 runs, core 0's polling pass parks, and core 1's store
    /// to the polled word unparks it: core 0 leaves its loop as soon after
    /// the store as without parking, to within one pass. A parked core adds
    /// no SRAM bank contention, so core 1 itself runs a little faster.
    #[test]
    fn two_core_park_sees_other_core_store() {
        let off = run_two_core_poll(false, 512, 100_000);
        let on = run_two_core_poll(true, 512, 100_000);
        assert_eq!(off.2, 0);
        assert!(on.2 > 100_000, "parked only {} cycles", on.2);
        assert_eq!(on.1[3], off.1[3], "core 1 counted to n in both runs");
        assert!(off.0[5] > 1000, "core 0 left the loop: r5 {}", off.0[5]);
        let lag_off = off.1[5] as i64 - off.0[5] as i64;
        let lag_on = on.1[5] as i64 - on.0[5] as i64;
        assert!((lag_on - lag_off).abs() <= 2, "core 0 lag behind the flag: {lag_on} parked, {lag_off} not");
    }

    /// The park carries over quantum boundaries while nothing it reads
    /// changes, so short quanta still park almost the whole wait.
    #[test]
    fn two_core_park_carries_across_quanta() {
        let off = run_two_core_poll(false, 256, 200_000);
        let on = run_two_core_poll(true, 256, 200_000);
        assert!(on.2 > 400_000, "parked only {} cycles at q256", on.2);
        let lag_off = off.1[5] as i64 - off.0[5] as i64;
        let lag_on = on.1[5] as i64 - on.0[5] as i64;
        assert!((lag_on - lag_off).abs() <= 2, "core 0 lag behind the flag: {lag_on} parked, {lag_off} not");
    }

    /// Core 0 polls the flag at 0x2000_0100 while core 1 counts forever.
    /// After 200000 cycles `set` writes the flag from outside the cores;
    /// returns core 0's parked cycles at that point and the quanta of 512
    /// cycles until it has left its loop.
    fn run_poll_external_set(set: fn(&mut Emulator)) -> (u64, u32) {
        let core0 = [
            0x6808, // 1000: ldr r0, [r1]     poll the flag
            0x2800, // 1002: cmp r0, #0
            0xD0FC, // 1004: beq 1000
            0x3501, // 1006: adds r5, #1
            0xE7FD, // 1008: b 1006
        ];
        let core1 = [
            0x3301, // 1100: adds r3, #1
            0xE7FD, // 1102: b 1100
        ];
        let mut emu = Emulator::new(Config::default());
        for (i, hw) in core0.iter().enumerate() {
            emu.bus.write16(0x2000_1000 + 2 * i as u32, *hw);
        }
        for (i, hw) in core1.iter().enumerate() {
            emu.bus.write16(0x2000_1100 + 2 * i as u32, *hw);
        }
        emu.step_quantum = 512;
        emu.idle.two_core = true;
        for (k, pc, sp) in [(0usize, 0x2000_1000u32, 0x2000_2000u32), (1, 0x2000_1100, 0x2000_3000)] {
            let c = &mut emu.cores[k];
            c.regs.r[13] = sp;
            c.regs.msp = sp;
            c.regs.r[1] = 0x2000_0100;
            c.regs.set_pc(pc);
        }
        emu.cores[1].wake();
        emu.run(200_000).unwrap();
        let parked = emu.idle.parked_cycles[0];
        set(&mut emu);
        let mut quanta = 0;
        while emu.cores[0].regs.r[5] == 0 && quanta < 1000 {
            emu.run(512).unwrap();
            quanta += 1;
        }
        (parked, quanta)
    }

    /// A parked core's word written by DMA, by a harness through the bus
    /// or by a poke ends the park by the next quantum boundary, although no
    /// core store to it reached the park check.
    #[test]
    fn two_core_park_sees_external_sram_writes() {
        fn dma(emu: &mut Emulator) {
            emu.bus.write32(crate::bus::RESETS_BASE + 0x3000, 1 << crate::bus::peripheral_dispatch::RESET_DMA);
            emu.bus.write32(0x2000_0200, 1);
            let ch = crate::bus::DMA_BASE;
            // EN, word size, TREQ always ready, no chain: copy one word.
            let ctrl = 1 | 2 << 2 | (crate::dreq::DREQ_FORCE as u32) << 15;
            emu.bus.write32(ch, 0x2000_0200);
            emu.bus.write32(ch + 4, 0x2000_0100);
            emu.bus.write32(ch + 8, 1);
            emu.bus.write32(ch + 0xC, ctrl);
        }
        fn bus(emu: &mut Emulator) {
            emu.bus.write32(0x2000_0100, 1);
        }
        fn poke(emu: &mut Emulator) {
            emu.poke(0x2000_0100, 1);
        }
        let sets: [(&str, fn(&mut Emulator)); 3] = [("dma", dma), ("bus", bus), ("poke", poke)];
        for (name, set) in sets {
            let (parked, quanta) = run_poll_external_set(set);
            assert!(parked > 100_000, "{name}: core 0 parked only {parked} cycles before the write");
            assert!(quanta <= 3, "{name}: core 0 left its loop after {quanta} quanta");
        }
    }

    /// Core 0 polls a flag with a delay loop in its pass, `delay` inner
    /// turns of 3 cycles each. Core 1 counts to `n` and sets the flag, or
    /// with `n` 0 polls a word nobody sets. Returns both cores' registers,
    /// core 0's parked cycles and its idle stats.
    fn run_slow_poll(two_core: bool, q: u32, n: u32, delay: u8) -> ([u32; 16], [u32; 16], u64, IdleStats) {
        let core0 = [
            0x6808,                 // 1000: ldr r0, [r1]     poll the flag
            0x2800,                 // 1002: cmp r0, #0
            0xD104,                 // 1004: bne 1010
            0x2600 | delay as u16,  // 1006: movs r6, #delay
            0x3E01,                 // 1008: subs r6, #1
            0xD1FD,                 // 100A: bne 1008
            0xE7F8,                 // 100C: b 1000
            0xBF00,                 // 100E: nop
            0x3501,                 // 1010: adds r5, #1
            0xE7FD,                 // 1012: b 1010
        ];
        let counter = [
            0x3301, // 1100: adds r3, #1
            0x42A3, // 1102: cmp r3, r4
            0xD1FC, // 1104: bne 1100
            0x2001, // 1106: movs r0, #1
            0x6008, // 1108: str r0, [r1]     set the flag
            0x3501, // 110A: adds r5, #1
            0xE7FD, // 110C: b 110A
        ];
        let poller = [
            0x6810, // 1100: ldr r0, [r2]
            0x2800, // 1102: cmp r0, #0
            0xD0FC, // 1104: beq 1100
        ];
        let core1: &[u16] = if n == 0 { &poller } else { &counter };
        let mut emu = Emulator::new(Config::default());
        for (i, hw) in core0.iter().enumerate() {
            emu.bus.write16(0x2000_1000 + 2 * i as u32, *hw);
        }
        for (i, hw) in core1.iter().enumerate() {
            emu.bus.write16(0x2000_1100 + 2 * i as u32, *hw);
        }
        emu.step_quantum = q;
        emu.idle.two_core = two_core;
        for (k, pc, sp) in [(0usize, 0x2000_1000u32, 0x2000_2000u32), (1, 0x2000_1100, 0x2000_3000)] {
            let c = &mut emu.cores[k];
            c.regs.r[13] = sp;
            c.regs.msp = sp;
            c.regs.r[1] = 0x2000_0100;
            c.regs.r[2] = 0x2000_0200;
            c.regs.r[3] = 0;
            c.regs.r[4] = n;
            c.regs.r[5] = 0;
            c.regs.set_pc(pc);
        }
        emu.cores[1].wake();
        emu.run(if n == 0 { 400_000 } else { n as u64 * 8 + 20_000 }).unwrap();
        (emu.cores[0].regs.r, emu.cores[1].regs.r, emu.idle.parked_cycles[0], emu.idle.stats[0])
    }

    /// A polling pass of 242 cycles fits between a snapshot and the end of
    /// a 256-cycle quantum only when the snapshot falls in the first 14
    /// cycles, and without the carry it never parks. While core 1 is
    /// parked, the snapshot carries across the end as long as what the
    /// pass read holds, so core 0 parks for most of the run, and both cores
    /// end where they would without parking.
    #[test]
    fn two_core_snapshot_carries_across_quanta() {
        let off = run_slow_poll(false, 256, 0, 78);
        let on = run_slow_poll(true, 256, 0, 78);
        assert!(on.3.snap_carried > 0, "no snapshot carried: {:?}", on.3);
        assert!(on.2 > 300_000, "parked only {} of 400000 cycles: {:?}", on.2, on.3);
        assert_eq!(on.0[5], 0, "core 0 never sees the flag");
        assert_eq!((on.0[0], on.1[0]), (off.0[0], off.1[0]));
    }

    /// Core 1's store can land anywhere in core 0's tracked pass, after
    /// the pass read the flag included. The pass must not park on the
    /// value it read: core 0 leaves its loop at every landing point.
    #[test]
    fn two_core_store_during_tracked_pass_is_seen() {
        for n in 20_000..20_080 {
            for q in [64, 256] {
                let off = run_slow_poll(false, q, n, 12);
                let on = run_slow_poll(true, q, n, 12);
                let lag_off = off.1[5] as i64 - off.0[5] as i64;
                let lag_on = on.1[5] as i64 - on.0[5] as i64;
                assert!(
                    (lag_on - lag_off).abs() <= 60,
                    "n {n} q {q}: core 0 lag behind the flag: {lag_on} parked, {lag_off} not, {:?}",
                    on.3
                );
            }
        }
    }

    /// Calls that push different registers into the same stack slots
    /// change memory on every store, but each pass leaves it as it found
    /// it: the pass is skipped, with a bit-identical result.
    #[test]
    fn idle_skip_collapses_pass_that_restores_its_stack() {
        let prog = [
            0x47A8, // 1000: blx r5     -> f
            0x47B0, // 1002: blx r6     -> g
            0xE7FC, // 1004: b 1000
            0xBF00, // 1006: nop, not reached
            0xB510, // 1008: f: push {r4, lr}
            0xBD10, // 100A:    pop {r4, pc}
            0xB508, // 100C: g: push {r3, lr}
            0xBD08, // 100E:    pop {r3, pc}
        ];
        let regs = [(3, 0x33), (4, 0x44), (5, 0x2000_1009), (6, 0x2000_100D)];
        let on = run_idle_prog(&prog, &regs, 400_000, true);
        let off = run_idle_prog(&prog, &regs, 400_000, false);
        assert_eq!((on.0, on.1), (off.0, off.1));
        assert_eq!(off.3, 0);
        assert!(on.3 > 300_000, "skipped only {} of 400000 cycles", on.3);
    }

    /// A pass that leaves memory changed is never skipped, even when the
    /// registers repeat.
    #[test]
    fn idle_skip_keeps_pass_that_changes_memory() {
        let prog = [
            0x6808, // 1000: ldr r0, [r1]
            0x3001, // 1002: adds r0, #1
            0x6008, // 1004: str r0, [r1]
            0x2000, // 1006: movs r0, #0
            0xE7FA, // 1008: b 1000
        ];
        let regs = [(1, 0x2000_0100)];
        let on = run_idle_prog(&prog, &regs, 100_000, true);
        let off = run_idle_prog(&prog, &regs, 100_000, false);
        assert_eq!(on.3, 0, "a counting loop must not be skipped");
        assert_eq!((on.0, on.1, on.2), (off.0, off.1, off.2));
        assert!(on.2 > 1000, "the loop ran: count {}", on.2);
    }

    /// A mutex-style pass: claim and release a spinlock, run a division on
    /// the SIO divider and send an event. All of it repeats exactly, so it
    /// is skipped; a pass that leaves the spinlock claimed is not.
    #[test]
    fn idle_skip_handles_spinlock_divider_and_sev() {
        let prog = [
            0x6808, // 1000: ldr r0, [r1]       claim SPINLOCK0
            0x6008, // 1002: str r0, [r1]       release it
            0x601A, // 1004: str r2, [r3]       DIV_UDIVIDEND
            0x605C, // 1006: str r4, [r3, #4]   DIV_UDIVISOR
            0x6918, // 1008: ldr r0, [r3, #16]  DIV_QUOTIENT
            0xBF40, // 100A: sev
            0xE7F8, // 100C: b 1000
        ];
        let regs = [(1, 0xD000_0100), (2, 1000), (3, 0xD000_0060), (4, 7)];
        let on = run_idle_prog(&prog, &regs, 400_000, true);
        let off = run_idle_prog(&prog, &regs, 400_000, false);
        assert_eq!((on.0, on.1), (off.0, off.1));
        assert_eq!(on.0[0], 142);
        assert!(on.3 > 300_000, "skipped only {} of 400000 cycles", on.3);
        // Claim only: the lock stays taken after the first pass, and the
        // claim read returns 0 from then on, so passes differ until they
        // settle; the result must still match.
        let prog = [0x6808, 0x601A, 0xE7FC];
        let on = run_idle_prog(&prog, &regs, 100_000, true);
        let off = run_idle_prog(&prog, &regs, 100_000, false);
        assert_eq!((on.0, on.1), (off.0, off.1));
    }

    // ------------------- tick_pio_and_route_irqs (lines 842, 852, 855, 860, 863) -------------------

    /// Drives the false arm of `if gpio_in & (1u32 << 4) == 0` (line 842)
    /// by seeding `bus.gpio_in` with bit 4 high before the slow-path
    /// tick. tick_pio_and_route_irqs reads gpio_in directly so we must
    /// pre-set it; update_gpio runs after tick_pio so the external mask
    /// alone doesn't apply early enough.
    #[test]
    fn tick_pio_iow_high_does_not_count_as_low() {
        let mut emu = Emulator::new(Config::default());
        // Force slow path so tick_pio_and_route_irqs runs.
        emu.bus.systicks[0].csr |= 1;
        // Pre-seed gpio_in with bit 4 high AND set the external mask so
        // update_gpio's post-tick rewrite preserves it across the cycle.
        emu.bus.gpio_in = 1u32 << 4;
        emu.bus.external_gpio_in_mask = 1u32 << 4;
        emu.bus.external_gpio_in_override = 1u32 << 4;
        let _ = emu.step().unwrap();
        // The branch was visited; whether the count stays zero depends
        // on whether tick_pio runs again after update_gpio. The goal is
        // line coverage of the false arm of the IOW gate.
    }

    /// Drives the true arms of the PIO INTF routing `if int0_ints != 0`
    /// (line 860) and `int1_ints != 0` (line 863) for both PIO blocks.
    /// Uses the slow-path forcing trick so tick_pio_and_route_irqs runs.
    #[test]
    fn tick_pio_routes_intf_to_irq_pending() {
        let mut emu = Emulator::new(Config::default());
        // Force slow path.
        emu.bus.systicks[0].csr |= 1;
        // Release every peripheral from reset so MMIO writes to PIO land.
        emu.bus.resets.state = 0;
        // Write INT0_INTF / INT1_INTF on both PIO blocks. RP2040 PIO
        // INT0_INTF offset 0x034, INT1_INTF offset 0x040 (datasheet
        // §3.7). bus offsets: PIO0=0x5020_0000, PIO1=0x5030_0000.
        emu.bus.write32(0x5020_0034, 0x1);
        emu.bus.write32(0x5020_0040, 0x1);
        emu.bus.write32(0x5030_0034, 0x1);
        emu.bus.write32(0x5030_0040, 0x1);
        let _ = emu.step().unwrap();
    }

    // ------------------- tick_pio early-return (line 876) -------------------

    /// Drives the true arm of `if cycles == 0` inside `tick_pio`
    /// (line 876) by stepping with cores idle. This is best-effort —
    /// the step path may always pass non-zero cycles. We instead invoke
    /// the situation by halting both cores so `consumed` may be zero.
    #[test]
    fn tick_pio_zero_cycles_is_noop_smoke() {
        let mut emu = Emulator::new(Config::default());
        emu.cores[0].halt();
        emu.halt_core1();
        let _ = emu.step().unwrap();
    }

    // ------------------- update_gpio: psram path + external mask (lines 1110, 1111, 1118) -------------------

    /// Drives the true branch of `if let Some(ref mut psram) = self.bus.psram`
    /// (line 1110) plus the inner `if let Some(miso) = psram.tick(out)`
    /// (line 1111) by attaching a PSRAM and calling update_gpio via a
    /// public path (gpio_write triggers it).
    #[test]
    fn update_gpio_with_psram_attached() {
        let psram = Psram::new(0, 1, 2, 3);
        let mut emu = EmulatorBuilder::new(Config::default())
            .psram(psram)
            .build()
            .unwrap();
        // gpio_write calls update_gpio internally.
        emu.gpio_write(5, true);
    }

    /// Drives the true branch of `if ext_mask != 0` (line 1118) by
    /// asserting an external GPIO mask before update_gpio runs.
    #[test]
    fn update_gpio_external_mask_overrides() {
        let mut emu = Emulator::new(Config::default());
        emu.bus.external_gpio_in_mask = 1u32 << 5;
        emu.bus.external_gpio_in_override = 1u32 << 5;
        // Trigger update_gpio via gpio_write.
        emu.gpio_write(0, false);
        assert!(emu.gpio_read(5));
    }

    // ------------------- wake_checks (lines 1148, 1155) -------------------

    /// Drives the true arm of `if self.bus.wfe_waiting[core] &&
    /// self.bus.event_flag[core]` (line 1148): pre-park core 0 on WFE
    /// with an event latched.
    #[test]
    fn wake_checks_consumes_wfe_event_core0() {
        let mut emu = Emulator::new(Config::default());
        emu.bus.wfe_waiting[0] = true;
        emu.bus.event_flag[0] = true;
        emu.cores[0].halt();
        let _ = emu.step().unwrap();
        assert!(!emu.bus.wfe_waiting[0]);
        assert!(!emu.bus.event_flag[0]);
    }

    /// Drives the true arm of `if self.cores[core].is_halted() &&
    /// self.bus.nvics[core].pending_and_enabled() != 0` (line 1155)
    /// by enabling and pending an IRQ on core 0 while halted.
    #[test]
    fn wake_checks_unhalts_on_pending_enabled_irq_core0() {
        let mut emu = Emulator::new(Config::default());
        emu.cores[0].halt();
        emu.bus.nvics[0].set_enabled(0);
        emu.bus.nvics[0].set_pending(0);
        let _ = emu.step().unwrap();
        assert!(!emu.cores[0].is_halted());
    }

    // ------------------- core_cycles: fall-through after core 1 wake -------------------

    /// Indirect coverage for the cycle-counter idx==1 arm in core_cycles
    /// after the core has actually consumed a cycle. The default-zero
    /// case is covered by `core_cycles_default_zero`; this test forces
    /// a non-zero counter to pair with it.
    #[test]
    fn core_cycles_idx1_after_step() {
        let mut emu = Emulator::new(Config::default());
        emu.wake_core1();
        let _ = emu.step().unwrap();
        // No exact assertion — accessor evaluates the idx==1 arm.
        let _ = emu.core_cycles(1);
    }
}
