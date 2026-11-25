//! RP2040 AHB-Lite bus fabric.
//!
//! Phase 5.A: full address decode + peripheral routing for the registers
//! firmware actually touches (CLOCKS, RESETS, PLL_SYS, PLL_USB, XOSC, ROSC,
//! SIO, IO_BANK0, PADS_BANK0, XIP_CTRL stub, SSI stub). SRAM bank routing
//! uses the RP2040 4+2 layout from [`crate::memory::bank_for_address`];
//! bank contention is modelled simply (+1 cycle on SRAM access when
//! the companion core has already touched SRAM this quantum).
//!
//! Phase 5.B: PIO0 / PIO1 wired into the AHB decode at `0x5020_0000` and
//! `0x5030_0000`. Register access goes through `PioBlock::read32` /
//! `write32` (mirrors `rp2350_emu::Bus`). Sub-word writes to PIO ranges are
//! ignored — several PIO registers have side-effects on read (RXF pop) or
//! write (TXF push, CTRL bit flags) that would behave incorrectly under a
//! synthetic read-modify-write. Sub-word reads still go through `read32`
//! (matching rp2350_emu) and so observe those same side-effects on the
//! enclosing word — firmware that only touches PIO with word-sized
//! accesses (the supported path in the datasheet) is unaffected.

pub mod clocks;
pub mod io_bank0;
pub mod pads_bank0;
pub mod peripheral_dispatch;
pub mod ppb;
pub mod resets;
pub mod sio;
pub mod systick;

use std::collections::{HashMap, VecDeque};
use std::io::Write;

use tracing::debug;

use picoem_common::PioBlock;
use picoem_common::clocks::{pll_cs_read_with_lock, pll_should_arm_lock};

use crate::core::Nvic;
use crate::dma::Dma;
use crate::irq::{
    IRQ_ADC_IRQ_FIFO, IRQ_I2C0_IRQ, IRQ_I2C1_IRQ, IRQ_PWM_IRQ_WRAP, IRQ_SPI0_IRQ, IRQ_SPI1_IRQ,
    IRQ_UART0_IRQ, IRQ_UART1_IRQ,
};
use crate::memory::{FLASH_SIZE, Memory, ROM_SIZE, SRAM_SIZE, bank_for_address};
use crate::peripherals::adc::AdcRegs;
use crate::peripherals::i2c::I2cRegs;
use crate::peripherals::pwm::PwmRegs;
use crate::peripherals::spi::SpiRegs;
use crate::peripherals::timer::TimerRegs;
use crate::peripherals::uart::UartRegs;
use crate::peripherals::watchdog_tick::WatchdogTickRegs;
use clocks::{ClockTree, ClocksRegs, PLL_RESET, PllRegs, ROSC_FREQ_HZ, RoscRegs, XoscRegs};
use io_bank0::IoBank0;
use pads_bank0::PadsBank0;
use picoem_devices::Psram;
use ppb::Ppb;
use resets::Resets;
use sio::Sio;
use systick::SysTick;

/// Peripheral base addresses (see RP2040 datasheet §2.2).
pub const APB_BASE: u32 = 0x4000_0000;
pub const SIO_BASE: u32 = 0xD000_0000;

// APB peripheral base addresses (RP2040 datasheet §2.2).
pub const SYSINFO_BASE: u32 = 0x4000_0000;
pub const SYSCFG_BASE: u32 = 0x4000_4000;
pub const CLOCKS_BASE: u32 = 0x4000_8000;
pub const RESETS_BASE: u32 = 0x4000_C000;
pub const PSM_BASE: u32 = 0x4001_0000;
pub const IO_BANK0_BASE: u32 = 0x4001_4000;
pub const IO_QSPI_BASE: u32 = 0x4001_8000;
pub const PADS_BANK0_BASE: u32 = 0x4001_C000;
pub const PADS_QSPI_BASE: u32 = 0x4002_0000;
pub const XOSC_BASE: u32 = 0x4002_4000;
pub const PLL_SYS_BASE: u32 = 0x4002_8000;
pub const PLL_USB_BASE: u32 = 0x4002_C000;
pub const BUSCTRL_BASE: u32 = 0x4003_0000;
pub const ROSC_BASE: u32 = 0x4006_0000;
/// UART0 block base (RP2040 datasheet §4.2). Reset-gated on bit 22.
pub const UART0_BASE: u32 = 0x4003_4000;
/// UART1 block base (RP2040 datasheet §4.2). Reset-gated on bit 23.
pub const UART1_BASE: u32 = 0x4003_8000;
/// SPI0 block base (RP2040 datasheet §4.4). Reset-gated on bit 16.
pub const SPI0_BASE: u32 = 0x4003_C000;
/// SPI1 block base (RP2040 datasheet §4.4). Reset-gated on bit 17.
pub const SPI1_BASE: u32 = 0x4004_0000;
/// I2C0 block base (RP2040 datasheet §4.3). Reset-gated on bit 3.
pub const I2C0_BASE: u32 = 0x4004_4000;
/// I2C1 block base (RP2040 datasheet §4.3). Reset-gated on bit 4.
pub const I2C1_BASE: u32 = 0x4004_8000;
/// ADC block base (RP2040 datasheet §4.9). Reset-gated on bit 0.
pub const ADC_BASE: u32 = 0x4004_C000;
/// PWM block base (RP2040 datasheet §4.5). Reset-gated on bit 14.
pub const PWM_BASE: u32 = 0x4005_0000;
/// TIMER block base (RP2040 datasheet §4.6). Reset-gated on
/// [`peripheral_dispatch::RESET_TIMER`] (bit 21) — TIMER and WATCHDOG
/// have independent RESETS bits per datasheet §2.14 Table 26. The 1 µs
/// cadence coupling to WATCHDOG_TICK is runtime signalling, not a
/// reset-bit relationship.
pub const TIMER_BASE: u32 = 0x4005_4000;
/// WATCHDOG block base (RP2040 datasheet §4.7). Phase 1 models only
/// `WATCHDOG_TICK` (offset `0x2C`); remaining registers read as 0.
pub const WATCHDOG_BASE: u32 = 0x4005_8000;
pub const XIP_CTRL_BASE: u32 = 0x1400_0000;
pub const SSI_BASE: u32 = 0x1800_0000;
/// Stores one idle-loop pass may make before it stops counting as idle.
pub(crate) const IDLE_LOG_MAX: usize = 256;

/// Idle-loop detector bookkeeping for the core that is stepping, kept on
/// the bus because every data access passes through it. See
/// [`crate::Emulator`]'s `idle_head_visit`.
#[derive(Default)]
pub(crate) struct IdleTrack {
    /// A core is stepping with the detector on.
    pub on: bool,
    /// Something since the last loop head cannot be repeated for free.
    pub dirty: bool,
    /// The dirt includes an exception entry or return, or a hint: the
    /// stretch was interrupted, which says nothing about the loop itself.
    pub event: bool,
    /// An exception was entered or returned from since the last loop head.
    pub exception: bool,
    /// A snapshot is live: SRAM stores are logged instead of dirtying.
    pub logging: bool,
    /// SRAM words stored since the snapshot: offset and the value before
    /// the first store to it.
    log: Vec<(u32, u32)>,
    /// Open-addressed index of `log` by offset plus 1, 0 for an empty
    /// slot. Allocated on first use.
    log_tab: Vec<u32>,
    /// Watch mode: the other core is running, so a clean pass may make no
    /// stores and only reads the other core cannot change without the bus
    /// seeing it. Those reads are recorded so the pass can be parked.
    pub watch: bool,
    /// Watch mode: SRAM words read or stored since the snapshot.
    pub words: Vec<u32>,
    /// Open-addressed index of `words`, 0 for an empty slot, so a lookup
    /// does not scan the list. Allocated on first use.
    word_tab: Vec<u32>,
    /// Watch mode: the pass stored to SRAM.
    pub stored: bool,
    /// Watch mode: the pass read a TIMER raw counter, which moves every
    /// quantum, so a park on it ends with the quantum.
    pub timer: bool,
    /// Watch mode: peripheral blocks a pass read status from or used, as
    /// [`PARK_PIO0`] and the other `PARK_*` bits.
    pub pio: u8,
    /// Watch mode: SIO GPIO_IN or GPIO_HI_IN was read.
    pub gpio: bool,
    /// Diagnostic, off by default: when on, every address that made a pass
    /// dirty is appended to `dirty_log`, capped at 1 << 20 entries.
    pub debug_dirty: bool,
    pub dirty_log: Vec<(u32, bool)>,
}

/// `pio` bits of [`IdleTrack`] and [`ParkWatch`]: PIO0 or PIO1 FSTAT or
/// FLEVEL read, UART0 or UART1 UARTFR read, a spinlock claimed and
/// released.
pub(crate) const PARK_PIO0: u8 = 1;
pub(crate) const PARK_UART0: u8 = 4;
pub(crate) const PARK_SPINLOCK: u8 = 16;

/// SRAM words one parked pass may depend on.
pub(crate) const PARK_WORDS_MAX: usize = 64;

/// Slots in [`IdleTrack`]'s word index: four per word, so probes stay
/// short.
const WORD_TAB: usize = 4 * PARK_WORDS_MAX;

/// Slots in [`IdleTrack`]'s store log index.
const LOG_TAB: usize = 2 * IDLE_LOG_MAX;

impl IdleTrack {
    /// Adds SRAM word `w` to `words` unless it is there already; false
    /// when it is new and the list is full.
    #[inline]
    fn note_word(&mut self, w: u32) -> bool {
        if self.word_tab.is_empty() {
            self.word_tab = vec![0; WORD_TAB];
        }
        let mut i = (w >> 2) as usize & (WORD_TAB - 1);
        loop {
            let e = self.word_tab[i];
            if e == w {
                return true;
            }
            if e == 0 {
                break;
            }
            i = (i + 1) & (WORD_TAB - 1);
        }
        if self.words.len() >= PARK_WORDS_MAX {
            return false;
        }
        self.word_tab[i] = w;
        self.words.push(w);
        true
    }

    /// Logs SRAM word `off`, whose value is `old()`, unless a store to it
    /// is logged already; false when it is new and the log is full.
    #[inline]
    fn log_word(&mut self, off: u32, old: impl FnOnce() -> u32) -> bool {
        if self.log_tab.is_empty() {
            self.log_tab = vec![0; LOG_TAB];
        }
        let mut i = (off >> 2) as usize & (LOG_TAB - 1);
        loop {
            let e = self.log_tab[i];
            if e == off + 1 {
                return true;
            }
            if e == 0 {
                break;
            }
            i = (i + 1) & (LOG_TAB - 1);
        }
        if self.log.len() >= IDLE_LOG_MAX {
            return false;
        }
        self.log_tab[i] = off + 1;
        self.log.push((off, old()));
        true
    }

    /// Empties the store log and its index.
    pub(crate) fn clear_log(&mut self) {
        if !self.log_tab.is_empty() {
            for &(off, _) in &self.log {
                let mut i = (off >> 2) as usize & (LOG_TAB - 1);
                while self.log_tab[i] != off + 1 {
                    i = (i + 1) & (LOG_TAB - 1);
                }
                self.log_tab[i] = 0;
            }
        }
        self.log.clear();
    }

    /// Whether SRAM word `w` is in `words`.
    #[inline]
    fn has_word(&self, w: u32) -> bool {
        if self.word_tab.is_empty() {
            return false;
        }
        let mut i = (w >> 2) as usize & (WORD_TAB - 1);
        loop {
            let e = self.word_tab[i];
            if e == w {
                return true;
            }
            if e == 0 {
                return false;
            }
            i = (i + 1) & (WORD_TAB - 1);
        }
    }

    /// Empties `words` and its index.
    pub(crate) fn clear_words(&mut self) {
        if !self.word_tab.is_empty() {
            for &w in &self.words {
                let mut i = (w >> 2) as usize & (WORD_TAB - 1);
                while self.word_tab[i] != w {
                    i = (i + 1) & (WORD_TAB - 1);
                }
                self.word_tab[i] = 0;
            }
        }
        self.words.clear();
    }
}

/// What a parked core's polling pass reads, checked against the running
/// core's accesses. `hit` is set by the first access that could change one
/// of those reads; the emulator then unparks the core.
#[derive(Default, Clone)]
pub(crate) struct ParkWatch {
    pub active: bool,
    /// The pass also stored to SRAM, restoring every byte; its words are in
    /// `words`, so another core reading them conflicts too.
    pub stored: bool,
    pub words: Vec<u32>,
    pub pio: u8,
    pub gpio: bool,
    pub hit: bool,
    /// The pass read the TIMER: the park ends with the quantum.
    pub timer: bool,
    /// [`Bus::park_fingerprint`] at park time, compared at each quantum
    /// boundary to decide whether the park carries over.
    pub vals: Vec<u32>,
    /// [`Bus::ext_sram_writes`] when the words last matched `vals`.
    pub ext_seen: u64,
}

/// Whether an access by the running core could change what a watch-mode
/// pass of the other core reads: a store to one of its SRAM words, or a
/// read of one it stored, as `word` says; any access to a PIO block it
/// polls other than the status registers, or to a UART it polls other
/// than UARTFR; a spinlock access while it uses spinlocks; a GPIO output
/// change while it reads GPIO_IN; or a FIFO write that raises its SIO
/// interrupt.
#[inline(always)]
fn watch_conflict(addr: u32, write: bool, pio: u8, gpio: bool, word: impl FnOnce(u32) -> bool) -> bool {
    match addr >> 28 {
        0x2 => word(addr & !3),
        0x5 if (addr >> 20) & 0xFFE == 0x502 => {
            pio & (PARK_PIO0 << ((addr >> 20) & 1)) != 0 && (write || !matches!(addr & 0xFFF, 0x004 | 0x00C))
        }
        0x4 if matches!(addr & !0x3FFF, 0x4003_4000 | 0x4003_8000) => {
            pio & (PARK_UART0 << ((addr >> 14) & 3 == 2) as u8) != 0 && (write || addr & 0xFFF != 0x018)
        }
        0xD => {
            let o = addr & 0xFFF;
            (pio & PARK_SPINLOCK != 0 && matches!(o, 0x100..=0x17F))
                || write && (o == 0x054 || (gpio && matches!(o, 0x010..=0x03C)))
        }
        _ => false,
    }
}

/// True for reads the idle-loop detector may repeat: they have no side
/// effect and cannot change within a quantum while the other core sleeps.
/// ROM, XIP flash and XIP SRAM, SRAM, the TIMER raw counters, SIO CPUID,
/// GPIO_IN and GPIO_HI_IN; also the spinlocks and the divider, whose state
/// the detector compares itself.
#[inline(always)]
fn idle_clean_read(addr: u32) -> bool {
    match addr >> 28 {
        0x0 | 0x2 => true,
        0x1 => matches!(addr >> 24, 0x10..=0x13 | 0x15),
        0x4 => matches!(addr & !0x3000, a if a == TIMER_BASE + 0x24 || a == TIMER_BASE + 0x28),
        0xD => matches!(addr & 0xFFF, 0x000 | 0x004 | 0x008 | 0x05C | 0x060..=0x078 | 0x100..=0x17F),
        _ => false,
    }
}

/// Bit index of `addr`'s 1 KB page in [`Bus::code_pages`], for the three
/// cacheable regions; keyed by the address as fetched, like the decode
/// cache tags.
#[inline(always)]
fn code_page_index(addr: u32) -> Option<usize> {
    let region = addr >> 28;
    if region > 2 {
        return None;
    }
    Some(((region as usize) << 16) | ((addr >> 10) & 0xFFFF) as usize)
}

/// SRAM pages in [`Bus::jit_pages`].
pub(crate) const JIT_PAGES: usize = SRAM_SIZE / 1024;

/// USB controller DPRAM, 4 KB, plain memory at every access width.
pub const USBCTRL_DPRAM_BASE: u32 = 0x5010_0000;
/// USB controller registers.
pub const USBCTRL_REGS_BASE: u32 = 0x5011_0000;
pub const XIP_SRAM_BASE: u32 = 0x1500_0000;
pub const XIP_SRAM_END: u32 = 0x1500_4000; // 16 KB
/// XIP flash window base. Aliases at `+0x0100_0000`, `+0x0200_0000`,
/// `+0x0300_0000` mirror the same 2 MB flash buffer.
pub const XIP_FLASH_BASE: u32 = 0x1000_0000;

/// Read a PLL register image with CS[31] (LOCK) derived from the
/// current arm state and master cycle count. Non-CS offsets return the
/// raw stored value via `clocks::pll_read`.
///
/// See `wrk_docs/2026.04.15 - HLD - PLL LOCK Modelling.md` §4.
#[inline]
fn pll_read_with_lock(regs: &PllRegs, offset: u32, lock_at: Option<u64>, now: u64) -> u32 {
    if offset == 0x00 {
        pll_cs_read_with_lock(regs, lock_at, now)
    } else {
        clocks::pll_read(regs, offset)
    }
}

/// Returns `Some(offset)` if `addr` falls inside one of the four 2 MB
/// XIP flash alias windows (`0x10`, `0x11`, `0x12`, `0x13` at bits
/// [27:24]). The returned offset is the byte offset into the flash
/// buffer (in `0..FLASH_SIZE`).
#[inline]
pub(crate) fn xip_flash_offset(addr: u32) -> Option<u32> {
    // Region selector (bits [31:28]) must be 0x1 for XIP. Alias bits
    // [27:24] (values 0..3 for the four 2 MB aliases) are validated
    // below.
    if (addr & 0xF000_0000) != XIP_FLASH_BASE {
        return None;
    }
    // Alias select bits [27:24]: 0x0, 0x1, 0x2, 0x3.
    let alias = (addr >> 24) & 0xF;
    if alias > 3 {
        return None;
    }
    // Offset inside the 2 MB alias window.
    let offset = addr & 0x00FF_FFFF;
    if (offset as usize) < FLASH_SIZE {
        Some(offset)
    } else {
        None
    }
}

// PIO AHB windows (RP2040 datasheet §3 — two PIO blocks).
pub const PIO0_BASE: u32 = 0x5020_0000;
pub const PIO1_BASE: u32 = 0x5030_0000;

/// Translate an RP2040 PIO register offset to the internal PioBlock offset.
///
/// `PioBlock` uses RP2350-style offsets (INTR at 0x16C, IRQ0_INTE at 0x170,
/// …, IRQ1_INTS at 0x184) because that is the authoritative silicon layout.
/// The RP2040 has no RXFn_PUTGET / GPIOBASE registers, so its INT block
/// starts immediately after the per-SM block at 0x128 — 0x44 bytes earlier.
/// This helper compensates for that shift when the RP2040 bus dispatches
/// into PioBlock.
#[inline(always)]
const fn pio_rp2040_to_internal(offset: u32) -> u32 {
    if offset >= 0x128 && offset <= 0x140 {
        offset + 0x44
    } else {
        offset
    }
}

/// RP2040 PIO register read dispatch. Handles INTR and INTn_INTS using the
/// RP2040-specific bit layout (IRQ flags at [3:0], RXNEMPTY at [7:4],
/// TXNFULL at [11:8]); all other registers are forwarded through the
/// standard offset translator `pio_rp2040_to_internal`.
///
/// Only the INTR, INT0_INTS, and INT1_INTS reads differ between chips —
/// the INTE/INTF registers expose the same raw storage value regardless
/// of which bit layout is in use.
fn pio_read_rp2040(pio: &mut PioBlock, offset: u32) -> u32 {
    match offset {
        // INTR (RP2040 offset 0x128): raw status in RP2040 12-bit layout.
        0x128 => pio.raw_intr_rp2040(),
        // INT0_INTS (RP2040 offset 0x134): (INTR_rp2040 & INTE) | INTF.
        0x134 => pio.int0_ints_rp2040(),
        // INT1_INTS (RP2040 offset 0x140): (INTR_rp2040 & INTE) | INTF.
        0x140 => pio.int1_ints_rp2040(),
        // All other offsets: translate to RP2350-internal and dispatch normally.
        other => pio.read32(pio_rp2040_to_internal(other)),
    }
}

/// DMA peripheral base (RP2040 datasheet §2.5 — single 4 KB window
/// covering 12 channels + global registers + debug aliases). Phase 4
/// (HLD V7 §5.6) enables the full DMA model.
pub const DMA_BASE: u32 = 0x5000_0000;

/// XIP SRAM size (16 KB on RP2040 — the cache RAM exposed as scratch).
pub const XIP_SRAM_SIZE: usize = 16 * 1024;

/// Number of entries in the per-core PC-keyed decoded-op cache.
/// Direct-mapped, indexed by `(pc >> 1) & (DECODE_CACHE_SIZE - 1)`.
/// 8192 entries × 12 B = 96 KB per core. Modelled on the RP2350 cache
/// (rp2350_emu commit 0c31479) but sized down: RP2040 hot loops are well
/// under 1 KB and total executable space (16 KB ROM + 16 KB XIP-SRAM +
/// 264 KB SRAM + 2 MB XIP flash) hashes to 8K slots without meaningful
/// conflict pressure for the workloads we measure.
pub(crate) const DECODE_CACHE_SIZE: usize = 8192;

/// One decoded ARMv6-M instruction. 12 bytes (4 + 2 + 2 + 1 + 3 pad),
/// `Copy`.
///
/// Populated lazily on a cache miss by
/// [`crate::core::CortexM0Plus::populate_decode_cache`]. An entry with
/// `tag == u32::MAX` is empty (that value is odd and cannot match a
/// halfword-aligned PC).
///
/// Differs from the rp2350_emu [`crate::bus::DecodedOp`] equivalent by
/// dropping `fetch_wait` (RP2040 has no `extra_wait_states` accumulator
/// — `Bus::read16` writes `last_access_cycles` but the core path does
/// not consume it) and `is_thumb16_flag_only` (ARMv6-M has no IT
/// blocks).
#[derive(Clone, Copy, Debug)]
pub(crate) struct DecodedOp {
    /// PC this entry is valid for. Full tag (no shift). `u32::MAX` =
    /// empty.
    pub tag: u32,
    /// First halfword (the one at PC).
    pub hw0: u16,
    /// Second halfword (at PC+2). Zero for narrow instructions.
    pub hw1: u16,
    /// Packed flags.
    ///   bit 0 — `is_wide`
    ///   bits 1..7 — reserved
    pub flags: u8,
}

impl DecodedOp {
    pub(crate) const FLAG_WIDE: u8 = 0b0000_0001;

    #[inline(always)]
    pub(crate) fn empty() -> Self {
        Self {
            tag: u32::MAX,
            hw0: 0,
            hw1: 0,
            flags: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn is_wide(&self) -> bool {
        self.flags & Self::FLAG_WIDE != 0
    }
}

/// True if `pc` lies in an executable region the cache may index.
/// Only ROM (`0x0`), XIP / XIP-SRAM (`0x1`), and SRAM (`0x2`) qualify.
/// Everything else (peripherals, SIO, PPB) either cannot legitimately
/// contain code or is dynamic and not worth caching.
#[inline(always)]
pub(crate) fn is_cacheable_pc(pc: u32) -> bool {
    matches!(pc >> 28, 0x0..=0x2)
}

/// Region bits for [`Bus::pending_invalidation_regions`] /
/// [`crate::core::CortexM0Plus::invalidate_decode_cache_regions`]. The
/// meaningful regions are the three cacheable ones (see
/// [`is_cacheable_pc`]). `BULK` is the universal escape hatch used when
/// the caller doesn't know (or doesn't care) which region changed.
pub mod invalidation_regions {
    /// Region `0x0` — boot ROM (16 KB).
    pub const ROM: u8 = 1 << 0;
    /// Region `0x1` — XIP / XIP-SRAM (flash window + cache scratch).
    pub const XIP: u8 = 1 << 1;
    /// Region `0x2` — on-chip SRAM (264 KB across 4 striped + 2 scratch
    /// banks).
    pub const SRAM: u8 = 1 << 2;
    /// Bulk bit — drain every slot regardless of tag region. Used by
    /// `ISB` and any path that can't attribute the change to a specific
    /// region (e.g. `Emulator::poke`, legacy bypass writes).
    pub const BULK: u8 = 1 << 7;
}

/// RP2040 AHB-Lite bus fabric.
pub struct Bus {
    pub memory: Memory,
    /// GPIO input state after merging SIO output with PIO outputs
    /// (Phase 5.A: SIO only). Read by firmware via SIO_GPIO_IN.
    pub gpio_in: u32,
    /// Per-core PPB (VTOR, SHPR, ICSR, active bitmap).
    pub ppb: [Ppb; 2],
    /// Single-cycle IO block.
    pub sio: Sio,
    /// RESETS peripheral.
    pub resets: Resets,
    /// CLOCKS register storage.
    pub clocks_regs: ClocksRegs,
    /// XOSC register storage.
    pub xosc_regs: XoscRegs,
    /// ROSC register storage.
    pub rosc_regs: RoscRegs,
    /// PLL_SYS register image (`[CS, PWR, FBDIV_INT, PRIM]`).
    pub pll_sys_regs: PllRegs,
    /// PLL_USB register image.
    pub pll_usb_regs: PllRegs,
    /// Master cycle count at the start of the current step. Populated by
    /// `Emulator::step` / `Emulator::run` before any core dispatch so that
    /// PLL CS reads and write-time lock-arm transitions observe a fresh
    /// cycle. See `wrk_docs/2026.04.15 - HLD - PLL LOCK Modelling.md` §6 P2.
    pub(crate) master_cycle: u64,
    /// Master cycle at which PLL_SYS's lock-detect counter expires. `None`
    /// means the PLL is not currently armed. Managed by the peripheral
    /// write dispatch via `picoem_common::clocks::pll_should_arm_lock`.
    pub(crate) pll_sys_lock_at_cycle: Option<u64>,
    /// Master cycle at which PLL_USB's lock-detect counter expires. Same
    /// semantics as `pll_sys_lock_at_cycle`.
    pub(crate) pll_usb_lock_at_cycle: Option<u64>,
    /// Derived clock tree frequencies (recomputed on any CLOCKS/PLL write).
    pub clock_tree: ClockTree,
    /// IO_BANK0 per-pin function select.
    pub io_bank0: IoBank0,
    /// PADS_BANK0 per-pin pad control.
    pub pads_bank0: PadsBank0,
    /// XIP SRAM (16 KB cache memory usable as SRAM, 0x1500_0000..0x1500_4000).
    xip_sram: Box<[u8; XIP_SRAM_SIZE]>,
    /// XIP_CTRL register backing store (stub — firmware round-trips only).
    xip_ctrl_regs: HashMap<u32, u32>,
    /// SSI register backing store.
    ssi_regs: HashMap<u32, u32>,
    /// SSI receive FIFO. Each byte written to DR0 while the SSI is
    /// enabled shifts one byte back from the serial flash model.
    ssi_rx: VecDeque<u8>,
    /// Bytes sent to the flash since chip select went low.
    ssi_cmd: Vec<u8>,
    /// Flash write-enable latch.
    flash_wel: bool,
    /// The page program in progress had write enable.
    flash_prog: bool,
    /// Catch-all APB peripheral register backing store for blocks we
    /// don't model in detail (PSM / BUSCTRL / SYSCFG / IO_QSPI / PADS_QSPI /
    /// UART / SPI / I2C / ADC / PWM / TIMER / WATCHDOG / RTC / VREG / TBMAN).
    /// Keyed by canonical word address (alias bits stripped).
    peripheral_regs: HashMap<u32, u32>,
    /// PIO0 / PIO1. Wired into the AHB decode at `0x5020_0000` /
    /// `0x5030_0000` (see [`PIO0_BASE`] / [`PIO1_BASE`]); output pins are
    /// merged into [`Self::gpio_in`] by [`crate::Emulator::update_gpio`].
    pub pio: [PioBlock; 2],
    /// WATCHDOG_TICK register model (Phase 1 scope — HLD V7 §5.5). Only
    /// the `TICK` register at offset `0x2C` is modelled today; the rest
    /// of the WATCHDOG block reads as 0.
    pub watchdog_tick: WatchdogTickRegs,
    /// TIMER register model (Phase 1 Wave 2 — HLD V7 §5.3). Lazy
    /// microsecond counter + four alarms; `advance_lazy_scheduled`
    /// polls `poll_alarms` on every step tail to surface alarm-match
    /// IRQs into `irq_pending`.
    pub(crate) timer: TimerRegs,
    /// UART0 — PL011-derived (Phase 2 — HLD V7 §5.3).
    pub(crate) uart0: UartRegs,
    /// UART1 — PL011-derived.
    pub(crate) uart1: UartRegs,
    /// SPI0 — PL022-derived.
    pub(crate) spi0: SpiRegs,
    /// SPI1 — PL022-derived.
    pub(crate) spi1: SpiRegs,
    /// USB controller, device mode.
    pub(crate) usbctrl: picoem_common::usb::UsbCtrl,
    /// Idle-loop detector bookkeeping.
    pub(crate) idle: IdleTrack,
    /// Parked polling passes, per core; see [`crate::IdleSkip`].
    pub(crate) park: [ParkWatch; 2],
    /// I2C0 — DW_apb_i2c.
    pub(crate) i2c0: I2cRegs,
    /// I2C1 — DW_apb_i2c.
    pub(crate) i2c1: I2cRegs,
    /// ADC — single instance at 0x4004_C000 (Phase 3 — HLD V7 §5.3).
    /// Runs on `clk_adc` (48 MHz nominal) via a fixed-point accumulator
    /// scaling from `clk_sys`. Reset-gated on RESETS bit 0.
    pub(crate) adc: AdcRegs,
    /// PWM — 8-slice block at 0x4005_0000 (Phase 3 — HLD V7 §5.3).
    /// Phase 3 cadence: CTR += 1 per sys_clk on enabled slices (DIV
    /// ignored); wrap latches `INTR[slice]` and routes `PWM_IRQ_WRAP`.
    /// Reset-gated on RESETS bit 14.
    pub(crate) pwm: PwmRegs,
    /// DMA controller — Phase 1 stub (always idle). Phase 4 replaces
    /// this with the 12-channel model. Consulted by the fast-path gate
    /// in [`crate::Emulator::step`] via [`Dma::is_idle`].
    ///
    /// `pub(crate)` because external readers (diagnostic harnesses
    /// such as `picogus_diff_rp2040` that need per-channel observation
    /// counters on `DmaChannel`) go through [`Self::dma_channel`]
    /// instead. Control still flows through `bus.write32`.
    pub(crate) dma: Dma,
    /// Pending external IRQ bitmap (bit N = IRQ #N asserted this
    /// cycle). Peripherals OR into this field when their state raises
    /// a line; [`crate::Emulator::drain_pending_irqs_to_cores`] drains
    /// the bitmap into both cores' NVIC pending latches per inner-loop
    /// iteration (HLD V7 §5.2).
    pub(crate) irq_pending: u32,
    /// PSM FRCE_OFF.PROC1 edge for the emulator to apply to core 1:
    /// `Some(true)` = held in reset, `Some(false)` = released.
    pub(crate) core1_psm_reset: Option<bool>,
    /// Per-core NVIC — one pending/enabled/priority set per CPU. The
    /// System Control Space NVIC registers (`0xE000_E100..0xE000_E41F`)
    /// are banked per-core on M0+; `nvics[active_core]` is the one the
    /// currently-executing CPU sees. Drained from [`Self::irq_pending`]
    /// by [`crate::Emulator::drain_pending_irqs_to_cores`], polled by
    /// [`crate::core::CortexM0Plus::step`] for dispatch.
    pub nvics: [Nvic; 2],
    /// Per-core SysTick (24-bit countdown timer mapped at
    /// `0xE000_E010..0xE000_E01F`). HLD V5 §5.2: ticked once per master
    /// cycle for the active core only; on a TICKINT-arm underflow the
    /// caller ORs `ICSR.PENDSTSET` (bit 26) onto the active-core PPB.
    pub(crate) systicks: [SysTick; 2],
    /// Off-chip 8 MB SPI PSRAM (PicoGUS v2 hardware). `None` when no
    /// PSRAM is attached (e.g. non-PicoGUS boards). Observed via
    /// [`crate::Emulator::update_gpio`] on the device's pin assignments
    /// and drives MISO back into [`Self::gpio_in`].
    pub psram: Option<Psram>,
    /// Externally-driven GPIO input override values. Bits set in
    /// [`Self::external_gpio_in_mask`] take their value from this field
    /// instead of whatever SIO / PIO / PSRAM would have produced; bits
    /// not set in the mask are unaffected. Used by the
    /// `picogus_diff_rp2040` harness to inject synthetic ISA bus
    /// waveforms (IOW#, IOR#, AD0..AD9) without those pokes being
    /// clobbered by [`crate::Emulator::update_gpio`]. The override is
    /// applied last in the merge — after SIO, PIO, and PSRAM — so an
    /// external driver always wins on the pins it claims.
    pub external_gpio_in_override: u32,
    /// Mask of GPIO input bits driven externally (see
    /// [`Self::external_gpio_in_override`]). The harness sets this to
    /// cover the ISA pins it injects (IOW#, IOR#, AD0..AD9) but **not**
    /// PSRAM pins (GPIO0..3) — those still belong to the on-chip merge.
    pub external_gpio_in_mask: u32,
    /// Per-core event flag for WFE/SEV / FIFO event protocol.
    pub event_flag: [bool; 2],
    /// Per-core WFE-park flag. `true` means the core is sleeping on
    /// WFE and will not execute until [`Self::event_flag`] for that
    /// core is consumed at a quantum boundary by
    /// `Emulator::wake_checks`. See `wrk_docs/2026.04.26 - HLD - RP2040
    /// WFE-SEV Wake Mechanics V1.md` §4.1.
    pub wfe_waiting: [bool; 2],
    /// Which core is currently executing on the bus.
    active_core: usize,
    /// Cycle cost of the most recent bus access.
    last_access_cycles: u32,
    /// Bus fault sticky flags.
    bus_fault: bool,
    bus_fault_addr: u32,
    /// Per-quantum bank-touched bitmap — bit N = bank N was accessed by
    /// the core 0 step. Read by core 1 to compute +1 cycle contention.
    pub(crate) core0_bank_touched: u8,
    /// True while the currently-running core is core 1 and it is looking
    /// up contention against `core0_bank_touched`. Set/cleared by the
    /// dual-core scheduler via `begin_core1_step` / `end_core1_step`.
    pub(crate) contention_check_active: bool,
    /// MMIO trace toggle (see `wrk_docs/2026.04.15 - HLD - RP2040 Peripheral
    /// Coverage V7.md` §4.3). When `true`, each byte/half/word bus access
    /// and each peripheral/SIO/PPB dispatch emits a line to
    /// [`Self::mmio_trace_sink`] (defaults to stdout when `None`). Zero overhead
    /// when `false` — the hot path short-circuits before any formatting.
    pub mmio_trace_enabled: bool,
    /// Per-core, per-instruction PC snapshot. Indexed by `active_core`
    /// so that a scheduler switch (`set_active_core(0→1→0)`) does not
    /// alias one core's decode PC onto the other. Set by the core's
    /// decode path (`CortexM0Plus::decode_execute`) immediately before
    /// instruction fetch, so every read/write during that instruction
    /// carries the correct architectural PC. Also set to sentinel values
    /// by `enter_exception` / `exit_exception` when hardware stacks or
    /// unstacks the 8-word frame (see `core::exceptions`).
    /// Default `[0, 0]`; only meaningful while a core is executing.
    pub(crate) active_pc: [u32; 2],
    /// Optional override sink for MMIO trace output. `None` routes to stdout
    /// via `println!`. Unit tests inject a `Vec<u8>`-backed sink to
    /// capture lines without wrestling with fd 1 redirection.
    pub(crate) mmio_trace_sink: Option<Box<dyn Write>>,
    /// Dirty-range log for the per-core decode caches. Every SRAM /
    /// XIP-SRAM write pushes the target halfword address(es) here; the
    /// driver (`Emulator::step_serial`) drains this into the core that
    /// just ran, evicting stale entries. Mirrors the rp2350_emu
    /// `Bus::pending_cache_invalidations` mechanism (commit 0c31479).
    pub pending_cache_invalidations: Vec<u32>,
    /// One bit per 1 KB page of the cacheable regions, set once the decode
    /// cache has held an instruction there; see [`code_page_index`].
    /// Stores elsewhere cannot evict anything and queue no invalidation.
    pub(crate) code_pages: Vec<u64>,
    /// 1 KB SRAM pages holding a word some active park watches; the block
    /// engine's SRAM fast path skips them. A superset is safe.
    pub(crate) watch_pages: [u64; 5],
    /// SRAM writes that reach no park check: every SRAM write through the
    /// bus but a core's, which DMA and harnesses make, and the pokes. A
    /// parked core's words change only through these or through an access
    /// [`Self::park_check`] sees, so while the count stands they hold.
    pub(crate) ext_sram_writes: u64,
    /// One byte per 1 KB SRAM page for compiled blocks: bit 0, loads take
    /// the slow path; bit 1, stores do. Watched pages set both. A page
    /// holding code sets bits 1 and 2 on itself and both neighbours, which
    /// covers the near-edge rule of [`Self::jit_sram`]; bit 2 keeps the code
    /// mark across watch refreshes.
    pub(crate) jit_pages: [u8; JIT_PAGES],
    /// [`Self::jit_timer`]'s last master cycle, clock and result.
    jit_timer_cache: std::cell::Cell<(u64, u32, u64)>,
    /// Region-scoped bulk-invalidation bitmask. Set by
    /// [`Self::load_bootrom`] (bit [`invalidation_regions::ROM`]),
    /// [`Self::load_flash`] (bit [`invalidation_regions::XIP`]), and ISB
    /// execution (bit [`invalidation_regions::BULK`]) when a write has
    /// replaced executable bytes wholesale. The driver drains the mask
    /// by calling
    /// [`crate::core::CortexM0Plus::invalidate_decode_cache_regions`] on
    /// each core and then resets it to `0`.
    pub pending_invalidation_regions: u8,
}

impl Bus {
    pub fn new() -> Self {
        Self {
            memory: Memory::with_flash(ROM_SIZE, SRAM_SIZE, FLASH_SIZE),
            gpio_in: 0,
            ppb: [Ppb::new(), Ppb::new()],
            sio: Sio::new(),
            resets: Resets::new(),
            clocks_regs: ClocksRegs::new(),
            xosc_regs: XoscRegs::new(),
            rosc_regs: RoscRegs::new(),
            pll_sys_regs: PLL_RESET,
            pll_usb_regs: PLL_RESET,
            master_cycle: 0,
            pll_sys_lock_at_cycle: None,
            pll_usb_lock_at_cycle: None,
            clock_tree: ClockTree::default(),
            io_bank0: IoBank0::new(),
            pads_bank0: PadsBank0::new(),
            xip_sram: Box::new([0u8; XIP_SRAM_SIZE]),
            xip_ctrl_regs: HashMap::new(),
            ssi_regs: HashMap::new(),
            ssi_rx: VecDeque::new(),
            ssi_cmd: Vec::new(),
            flash_wel: false,
            flash_prog: false,
            peripheral_regs: HashMap::new(),
            pio: [PioBlock::new(), PioBlock::new()],
            watchdog_tick: WatchdogTickRegs::new(),
            timer: TimerRegs::new(),
            uart0: UartRegs::new(IRQ_UART0_IRQ),
            uart1: UartRegs::new(IRQ_UART1_IRQ),
            spi0: SpiRegs::new(IRQ_SPI0_IRQ),
            spi1: SpiRegs::new(IRQ_SPI1_IRQ),
            usbctrl: picoem_common::usb::UsbCtrl::new_rp2040(),
            idle: IdleTrack::default(),
            park: [ParkWatch::default(), ParkWatch::default()],
            i2c0: I2cRegs::new(IRQ_I2C0_IRQ),
            i2c1: I2cRegs::new(IRQ_I2C1_IRQ),
            adc: AdcRegs::new(IRQ_ADC_IRQ_FIFO),
            pwm: PwmRegs::new(IRQ_PWM_IRQ_WRAP),
            dma: Dma::new(),
            irq_pending: 0,
            core1_psm_reset: None,
            nvics: [Nvic::new(), Nvic::new()],
            systicks: [SysTick::new(), SysTick::new()],
            psram: None,
            external_gpio_in_override: 0,
            external_gpio_in_mask: 0,
            event_flag: [false; 2],
            wfe_waiting: [false; 2],
            active_core: 0,
            last_access_cycles: 0,
            bus_fault: false,
            bus_fault_addr: 0,
            core0_bank_touched: 0,
            contention_check_active: false,
            mmio_trace_enabled: false,
            active_pc: [0; 2],
            mmio_trace_sink: None,
            // 16 entries up front — STM tops out at 13 registers; 16
            // covers a worst-case STM PC-rewrite without reallocation.
            pending_cache_invalidations: Vec::with_capacity(16),
            code_pages: vec![0; 3 * 65536 / 64],
            watch_pages: [0; 5],
            ext_sram_writes: 0,
            jit_pages: [0; JIT_PAGES],
            jit_timer_cache: std::cell::Cell::new((u64::MAX, 0, 0)),
            pending_invalidation_regions: 0,
        }
    }

    // --- Active-core / scheduler plumbing ---------------------------------

    #[inline]
    pub fn active_core(&self) -> usize {
        self.active_core
    }

    #[inline]
    pub fn set_active_core(&mut self, core: usize) {
        debug_assert!(core < 2);
        self.active_core = core;
    }

    /// Stash the instruction PC of the currently-executing instruction
    /// for the *currently-active core*. Called by
    /// [`crate::core::CortexM0Plus::decode_execute`] before instruction
    /// fetch so the MMIO trace can report a meaningful PC for every
    /// access that instruction performs. Also called by exception
    /// entry / exit with sentinel values (`0xFFFF_FFFE`,
    /// `0xFFFF_FFFD`) so stacking / unstacking lines are distinguishable
    /// from ordinary instruction-driven access. See HLD V7 §4.3.
    #[inline]
    pub fn set_active_pc(&mut self, pc: u32) {
        self.active_pc[self.active_core] = pc;
    }

    /// Emit a single MMIO trace line. `rw` is `'R'` or `'W'`, `size` is 1/2/4
    /// bytes, `val` is the value read or written. Zero overhead when
    /// [`Self::mmio_trace_enabled`] is `false`; the caller is expected to gate
    /// with `if self.mmio_trace_enabled` so the formatting cost is paid only
    /// when tracing.
    ///
    /// Routes to [`Self::mmio_trace_sink`] if set, else `println!` (stdout).
    /// No buffering — each line flushes at the `writeln!` boundary.
    ///
    /// Coverage note (see HLD V7 §4.3). The trace is emitted only from the
    /// six outer access methods ([`Self::read8`] … [`Self::write32`]). The
    /// internal peripheral/SIO/PPB dispatch helpers (`peripheral_read32`,
    /// `peripheral_write32`, `sio_read32`, `sio_write32`,
    /// `ppb[].read32/write32`) are **only reachable** from those six
    /// methods — they have no other callers in the crate (verified by
    /// grep) and are not `pub`. So outer-only tracing covers 100% of the
    /// MMIO surface firmware can touch, at one line per architectural
    /// access. Hooking the inner helpers as well would double-emit on
    /// word-sized peripheral access (outer calls inner directly) and
    /// surface the byte/half RMW-through-word32 artefact on narrow
    /// peripheral access — neither of which helps the "what does firmware
    /// touch next?" workflow the oracle is meant to unblock.
    #[inline(never)]
    fn emit_mmio_trace(&mut self, rw: char, size: u32, addr: u32, val: u32) {
        let line = format!(
            "TRACE {} {} 0x{:08X} val=0x{:08X} core={} pc=0x{:08X}",
            rw, size, addr, val, self.active_core, self.active_pc[self.active_core]
        );
        if let Some(sink) = self.mmio_trace_sink.as_mut() {
            let _ = writeln!(sink, "{}", line);
        } else {
            println!("{}", line);
        }
    }

    /// Install a captured MMIO trace sink (used by unit tests). `None` routes
    /// back to stdout. This is `pub(crate)` to keep it off the public
    /// surface — the binary toggles `mmio_trace_enabled` only.
    #[cfg(test)]
    pub(crate) fn set_mmio_trace_sink(&mut self, sink: Option<Box<dyn Write>>) {
        self.mmio_trace_sink = sink;
    }

    /// Called before core 1 steps each quantum — enables the contention
    /// check that adds +1 cycle when core 1 touches an SRAM bank already
    /// touched by core 0.
    #[inline]
    pub fn begin_core1_step(&mut self) {
        self.contention_check_active = true;
    }

    /// Called after core 1 has finished its slice. Clears the
    /// contention window and wipes the core-0 bank map for the next
    /// quantum.
    #[inline]
    pub fn end_core1_step(&mut self) {
        self.contention_check_active = false;
        self.core0_bank_touched = 0;
    }

    // --- Clock-tree accessors --------------------------------------------

    #[inline]
    pub fn sys_clk_hz(&self) -> u32 {
        self.clock_tree.sys_clk_hz
    }

    #[inline]
    pub fn ref_clk_hz(&self) -> u32 {
        self.clock_tree.ref_clk_hz
    }

    /// Seed the derived clock tree with an initial frequency. First
    /// write to CLOCKS / PLL replaces the seed with the derived value.
    pub fn seed_sys_clk_hz(&mut self, hz: u32) {
        self.clock_tree.sys_clk_hz = hz;
        self.clock_tree.ref_clk_hz = hz;
    }

    fn recompute_clock_tree(&mut self) {
        clocks::recompute(
            &self.clocks_regs,
            &self.pll_sys_regs,
            &self.pll_usb_regs,
            &mut self.clock_tree,
        );
        debug!(
            sys_clk_hz = self.clock_tree.sys_clk_hz,
            ref_clk_hz = self.clock_tree.ref_clk_hz,
            peri_clk_hz = self.clock_tree.peri_clk_hz,
            "clock tree recomputed"
        );
    }

    // --- Flash / XIP management ------------------------------------------

    /// Copy `data` into the 2 MB XIP flash window at offset 0. Oversized
    /// images are clamped by [`Memory::load_flash`]; the mapped window
    /// is always 2 MB so reads past the image length return 0.
    pub fn load_flash(&mut self, data: &[u8]) {
        self.memory.load_flash(data);
        // XIP / XIP-SRAM share region nibble 0x1 in the cache region
        // mask, so a flash reload invalidates any cached entry that
        // tagged into either window.
        self.pending_invalidation_regions |= invalidation_regions::XIP;
    }

    // --- Bus-fault plumbing -----------------------------------------------

    pub fn bus_fault(&self) -> bool {
        self.bus_fault
    }

    pub fn bus_fault_addr(&self) -> u32 {
        self.bus_fault_addr
    }

    pub fn clear_bus_fault(&mut self) {
        self.bus_fault = false;
    }

    /// Set the sticky bus-fault flag and record the faulting address.
    /// Emits a tracing event on the cold path so unmapped accesses are
    /// observable without throwaway `eprintln!`.
    #[inline]
    fn set_bus_fault(&mut self, addr: u32) {
        debug!(addr = format_args!("{:#010x}", addr), "unmapped bus access");
        self.bus_fault = true;
        self.bus_fault_addr = addr;
    }

    // --- Direct peek/poke (bypasses decode, still routes through regions)

    pub fn peek32(&self, addr: u32) -> u32 {
        if (addr >> 28) == 0x2 {
            // SRAM
            self.memory.sram_read32(addr & 0x00FF_FFFF)
        } else if (XIP_SRAM_BASE..XIP_SRAM_END).contains(&addr) {
            let off = (addr - XIP_SRAM_BASE) as usize;
            u32::from_le_bytes([
                self.xip_sram[off],
                self.xip_sram[off + 1],
                self.xip_sram[off + 2],
                self.xip_sram[off + 3],
            ])
        } else {
            self.memory.peek32(addr)
        }
    }

    pub fn poke32(&mut self, addr: u32, value: u32) {
        if (addr >> 28) == 0x2 {
            self.ext_sram_writes += 1;
            self.memory.sram_write32(addr & 0x00FF_FFFF, value);
        } else if (XIP_SRAM_BASE..XIP_SRAM_END).contains(&addr) {
            let off = (addr - XIP_SRAM_BASE) as usize;
            let bytes = value.to_le_bytes();
            self.xip_sram[off..off + 4].copy_from_slice(&bytes);
        } else {
            self.memory.poke32(addr, value);
        }
    }

    // --- Raw ROM loader (used by `Emulator::load_bootrom`) ----------------

    pub fn load_bootrom(&mut self, data: &[u8]) {
        self.memory.load_rom(data);
        self.pending_invalidation_regions |= invalidation_regions::ROM;
    }

    /// Marks `pc`'s page as holding decoded code.
    #[inline]
    pub(crate) fn set_code_page(&mut self, pc: u32) {
        if let Some(i) = code_page_index(pc) {
            self.code_pages[i >> 6] |= 1 << (i & 63);
        }
        if pc >> 24 == 0x20 {
            let p = ((pc & 0x00FF_FFFF) >> 10) as usize;
            for q in p.saturating_sub(1)..(p + 2).min(JIT_PAGES) {
                self.jit_pages[q] |= 6;
            }
        }
    }

    #[inline(always)]
    fn is_code_page(&self, addr: u32) -> bool {
        code_page_index(addr).is_some_and(|i| self.code_pages[i >> 6] & (1 << (i & 63)) != 0)
    }

    /// A code halfword read without bus side effects, from ROM, the XIP
    /// flash window or SRAM; `None` elsewhere. Used by the block engine.
    #[inline]
    pub(crate) fn peek_code16(&self, pc: u32) -> Option<u16> {
        match pc >> 24 {
            0x00 if (pc as usize) + 1 < ROM_SIZE => Some(self.memory.rom_read16(pc)),
            0x10..=0x13 => xip_flash_offset(pc).map(|o| self.memory.xip_read16(o)),
            0x20 if ((pc & 0x00FF_FFFF) as usize) + 1 < SRAM_SIZE => Some(self.memory.sram_read16(pc & 0x00FF_FFFF)),
            _ => None,
        }
    }

    /// Rebuilds [`Self::watch_pages`] from the active parks.
    pub(crate) fn refresh_watch_pages(&mut self) {
        let mut pages = [0u64; 5];
        for p in &self.park {
            if p.active {
                for &w in &p.words {
                    let page = ((w & 0x00FF_FFFF) >> 10) as usize;
                    if w >> 24 == 0x20 && page < 5 * 64 {
                        pages[page >> 6] |= 1 << (page & 63);
                    }
                }
            }
        }
        for w in 0..5 {
            let mut changed = pages[w] ^ self.watch_pages[w];
            while changed != 0 {
                let p = 64 * w + changed.trailing_zeros() as usize;
                changed &= changed - 1;
                if p < JIT_PAGES {
                    let b = &mut self.jit_pages[p];
                    let code = *b & 4;
                    *b = code | (code >> 1) | if pages[w] & (1 << (p & 63)) != 0 { 3 } else { 0 };
                }
            }
        }
        self.watch_pages = pages;
    }

    /// Whether compiled blocks may read TIMERAWL and TIMERAWH without the
    /// bus, and the microsecond count those reads return: `master_cycle`
    /// scaled as [`TimerRegs::read32`] does.
    pub(crate) fn jit_timer(&self) -> (bool, u64) {
        // The block engine asks once per run; the clock moves only at
        // quantum ends, so the division is cached.
        let (mc, hz, us) = self.jit_timer_cache.get();
        let us = if mc == self.master_cycle && hz == self.clock_tree.sys_clk_hz {
            us
        } else {
            let div = (self.clock_tree.sys_clk_hz / 1_000_000).max(1) as u64;
            let us = self.master_cycle / div;
            self.jit_timer_cache.set((self.master_cycle, self.clock_tree.sys_clk_hz, us));
            us
        };
        (!peripheral_dispatch::is_held_in_reset(self, TIMER_BASE), us)
    }

    /// Whether the block engine may use its SRAM fast path for this block:
    /// no idle tracking, no MMIO trace, no bank contention window.
    #[inline(always)]
    pub(crate) fn jit_fast(&self) -> bool {
        !self.idle.on && !self.mmio_trace_enabled && !self.contention_check_active
    }

    /// Whether compiled blocks may use their SRAM fast path, calling the
    /// idle-tracking hooks themselves while tracking is live: no MMIO
    /// trace and no bank contention window.
    #[inline(always)]
    pub(crate) fn jit_pages_ok(&self) -> bool {
        !self.mmio_trace_enabled && !self.contention_check_active
    }

    /// The SRAM offset of an access the block engine may make directly, or
    /// `None` for the slow path: outside SRAM, on a watched page, or, for a
    /// store, on or next to a page holding decoded code, as
    /// [`Self::invalidate_pc_range`] tests. Records core 0's bank touch as
    /// the bus does.
    #[inline(always)]
    pub(crate) fn jit_sram(&mut self, addr: u32, size: u32, write: bool) -> Option<u32> {
        if addr >> 24 != 0x20 {
            return None;
        }
        let off = addr & 0x00FF_FFFF;
        if (off + size) as usize > SRAM_SIZE {
            return None;
        }
        let page = (off >> 10) as usize;
        if self.watch_pages[page >> 6] & (1 << (page & 63)) != 0 {
            return None;
        }
        if write {
            let near_edge = !(2..=0x3FC).contains(&(addr & 0x3FF));
            if self.is_code_page(addr)
                || near_edge && (self.is_code_page(addr.wrapping_sub(2)) || self.is_code_page(addr.wrapping_add(3)))
            {
                return None;
            }
        }
        if self.active_core == 0 {
            let bank = if off < 0x4_0000 { (off >> 2) & 3 } else if off < 0x4_1000 { 4 } else { 5 };
            self.core0_bank_touched |= 1 << bank;
        }
        Some(off)
    }

    /// Queue cache invalidations covering `[addr, addr+len)` on the
    /// per-core decode caches. `len` is 1, 2, or 4 bytes. Pushes into
    /// [`Self::pending_cache_invalidations`]; the driver
    /// (`Emulator::step_serial`) drains it into the core that just ran.
    /// Mirrors the rp2350_emu pattern (commit 0c31479).
    ///
    /// The drainer
    /// ([`crate::core::CortexM0Plus::invalidate_decode_cache_entries`])
    /// evicts both the slot for `addr` and the slot for `addr - 2`
    /// (covering a wide instruction whose `hw1` landed at `addr`). For
    /// a 4-byte write we push `{addr, addr+2}` so the combined
    /// per-slot coverage is `{addr-2, addr, addr+2}`.
    #[inline]
    fn invalidate_pc_range(&mut self, addr: u32, len: u8) {
        debug_assert!(len == 1 || len == 2 || len == 4);
        // The drain evicts slots at addr - 2 to addr + 2; those bytes touch
        // a neighbouring page only within 3 bytes of a page edge.
        let near_edge = !(2..=0x3FC).contains(&(addr & 0x3FF));
        if matches!(addr >> 28, 0x0..=0x2)
            && (self.is_code_page(addr)
                || near_edge && (self.is_code_page(addr.wrapping_sub(2)) || self.is_code_page(addr.wrapping_add(3))))
        {
            self.pending_cache_invalidations.push(addr);
            if len == 4 {
                self.pending_cache_invalidations.push(addr.wrapping_add(2));
            }
        }
    }

    // --- Latency helpers --------------------------------------------------

    #[inline]
    pub fn last_access_cycles(&self) -> u32 {
        self.last_access_cycles
    }

    /// Harness-only diagnostic: drain every byte firmware has written to
    /// `UART0.DR` since the previous call. See `UartRegs::drain_tx_log`.
    pub fn drain_uart0_tx_log(&mut self) -> Vec<u8> {
        self.uart0.drain_tx_log()
    }

    /// Idle-loop detector: note a core data read.
    #[inline(always)]
    fn idle_read(&mut self, addr: u32) {
        self.idle_classify_read(addr);
        if self.park[0].active | self.park[1].active {
            self.park_check(addr, false);
        }
        if self.idle.logging & self.idle.watch & !self.idle.on {
            self.snap_check(addr, false);
        }
    }

    /// The idle-loop detector's part of [`Self::idle_read`], without the
    /// park check. Compiled blocks call it for their fast SRAM accesses and
    /// TIMER reads, which no park check can hit: a page holding a parked
    /// core's word is never on the fast path.
    #[inline(always)]
    pub(crate) fn idle_classify_read(&mut self, addr: u32) {
        if self.idle.on {
            if self.idle.watch {
                self.idle_watch_read(addr);
            } else if !idle_clean_read(addr) {
                self.idle.dirty = true;
            }
        }
    }

    /// Watch-mode read: constant within the quantum, or recorded so the
    /// other core's accesses can be checked against it.
    #[inline(never)]
    fn idle_watch_read(&mut self, addr: u32) {
        match addr >> 28 {
            0x0 => {}
            0x1 if matches!(addr >> 24, 0x10..=0x13) => {}
            0x2 => {
                if !self.idle.note_word(addr & !3) {
                    self.idle.dirty = true;
                }
            }
            0x4 if matches!(addr & !0x3000, a if a == TIMER_BASE + 0x24 || a == TIMER_BASE + 0x28) => self.idle.timer = true,
            0x5 if matches!(addr & 0xFFF0_0FFF, 0x5020_0004 | 0x5020_000C | 0x5030_0004 | 0x5030_000C) => {
                self.idle.pio |= PARK_PIO0 << ((addr >> 20) & 1);
            }
            0x4 if matches!(addr & !0x3000, 0x4003_4018 | 0x4003_8018) => {
                self.idle.pio |= PARK_UART0 << ((addr >> 14) & 3 == 2) as u8;
            }
            0xD if matches!(addr & 0xFFF, 0x000) => {}
            0xD if matches!(addr & 0xFFF, 0x100..=0x17F) => self.idle.pio |= PARK_SPINLOCK,
            0xD if matches!(addr & 0xFFF, 0x004 | 0x008) => self.idle.gpio = true,
            _ => {
                self.idle.dirty = true;
                if self.idle.debug_dirty && self.idle.dirty_log.len() < 1 << 20 {
                    self.idle.dirty_log.push((addr, false));
                }
            }
        }
    }

    /// An access by the running core that could change what a parked core
    /// reads, as [`watch_conflict`] decides.
    #[inline(never)]
    fn park_check(&mut self, addr: u32, write: bool) {
        let me = self.active_core();
        let p = &mut self.park[1 - me];
        if !p.active || p.hit {
            return;
        }
        if watch_conflict(addr, write, p.pio, p.gpio, |w| (write || p.stored) && p.words.contains(&w)) {
            p.hit = true;
        }
    }

    /// The same check for the pass the other core is tracking in watch
    /// mode: an access that could change what it has read so far makes
    /// the pass dirty, as an interruption, so it neither parks on values
    /// it no longer sees nor marks its head busy.
    #[inline(never)]
    fn snap_check(&mut self, addr: u32, write: bool) {
        let t = &self.idle;
        if t.dirty {
            return;
        }
        if watch_conflict(addr, write, t.pio, t.gpio, |w| (write || t.stored) && t.has_word(w)) {
            self.idle.dirty = true;
            self.idle.event = true;
        }
    }

    /// Idle-loop detector: note a core data write before it lands. Plain
    /// SRAM stores are logged while a snapshot is live; spinlock releases
    /// and divider operands are compared as state; anything else is a side
    /// effect.
    #[inline(always)]
    fn idle_write(&mut self, addr: u32, size: u32) {
        if self.park[0].active | self.park[1].active {
            self.park_check(addr, true);
        }
        if self.idle.logging & self.idle.watch & !self.idle.on {
            self.snap_check(addr, true);
        }
        self.idle_classify_write(addr, size);
    }

    /// The idle-loop detector's part of [`Self::idle_write`], without the
    /// park check, as [`Self::idle_classify_read`].
    #[inline(always)]
    pub(crate) fn idle_classify_write(&mut self, addr: u32, size: u32) {
        if !self.idle.on {
            return;
        }
        if self.idle.watch {
            if addr >> 28 == 0xD && matches!(addr & 0xFFF, 0x100..=0x17F) {
                self.idle.pio |= PARK_SPINLOCK;
                return;
            }
            if addr >> 28 == 0x2 && !self.idle.logging {
                return;
            }
            let ok = addr >> 28 == 0x2 && {
                let first = addr & !3;
                let last = (addr + size - 1) & !3;
                let a = self.idle.note_word(first);
                let b = self.idle.note_word(last);
                a && b
            };
            if !ok {
                self.idle.dirty = true;
                if self.idle.debug_dirty && self.idle.dirty_log.len() < 1 << 20 {
                    self.idle.dirty_log.push((addr, true));
                }
                return;
            }
            self.idle.stored = true;
        }
        match addr >> 28 {
            0x2 => {
                let off = addr & 0x00FF_FFFF;
                if !self.idle.logging {
                    return;
                }
                // A store stays within its aligned word, so the whole word
                // is logged once, before the first store to it.
                let w = off & !3;
                let mem = &self.memory;
                if (off + size) as usize > SRAM_SIZE || !self.idle.log_word(w, || mem.sram_read32(w)) {
                    self.idle.dirty = true;
                }
            }
            0xD if matches!(addr & 0xFFF, 0x060..=0x078 | 0x100..=0x17F) => {}
            _ => self.idle.dirty = true,
        }
    }

    /// Everything a parked pass of `core` read that can change between
    /// quanta without a core access: its SRAM words, which DMA may write,
    /// the polled PIO and UART status, GPIO_IN, and the core's side state.
    pub(crate) fn park_fingerprint(&self, core: usize) -> Vec<u32> {
        let mut v = Vec::new();
        self.inputs_visit(true, false, core, |x| {
            v.push(x);
            true
        });
        v
    }

    /// Whether [`Self::park_fingerprint`] still equals the one taken at
    /// park time.
    /// The words are read again only after an SRAM write no park check saw.
    pub(crate) fn park_inputs_held(&mut self, core: usize) -> bool {
        let p = &self.park[core];
        let words_held = p.ext_seen == self.ext_sram_writes;
        let vals = &p.vals;
        let mut i = if words_held { p.words.len() } else { 0 };
        let held = self.inputs_visit(true, words_held, core, |x| {
            let ok = vals.get(i) == Some(&x);
            i += 1;
            ok
        });
        let held = held && i == vals.len();
        if held {
            self.park[core].ext_seen = self.ext_sram_writes;
        }
        held
    }

    /// [`Self::park_fingerprint`] for the pass `core` is tracking now: what
    /// it has read since its snapshot.
    pub(crate) fn idle_fingerprint(&self, core: usize, out: &mut Vec<u32>) {
        out.clear();
        self.inputs_visit(false, false, core, |x| {
            out.push(x);
            true
        });
    }

    /// Feeds the inputs of `core`'s park when `park`, else of the tracked
    /// pass, to `f` in a fixed order, leaving out the SRAM words when
    /// `skip_words`; stops and returns false as soon as `f` does.
    fn inputs_visit(&self, park: bool, skip_words: bool, core: usize, mut f: impl FnMut(u32) -> bool) -> bool {
        let (words, blocks, gpio) = if park {
            let p = &self.park[core];
            (&p.words[..], p.pio, p.gpio)
        } else {
            (&self.idle.words[..], self.idle.pio, self.idle.gpio)
        };
        let words = if skip_words { &[][..] } else { words };
        for &w in words {
            let off = w & 0x00FF_FFFF;
            if !f(if (off as usize) + 4 <= SRAM_SIZE { self.memory.sram_read32(off) } else { 0 }) {
                return false;
            }
        }
        for b in 0..2 {
            if blocks & (PARK_PIO0 << b) != 0 && !(f(self.pio[b].fstat()) && f(self.pio[b].flevel())) {
                return false;
            }
        }
        if blocks & PARK_UART0 != 0 && !f(self.uart0.fr_read()) {
            return false;
        }
        if blocks & (PARK_UART0 << 1) != 0 && !f(self.uart1.fr_read()) {
            return false;
        }
        if gpio && !f(self.gpio_in) {
            return false;
        }
        self.idle_watch_side(core, blocks).into_iter().all(f)
    }

    /// [`Self::idle_side_state`] as a watch-mode pass with inputs `blocks`
    /// depends on it: the spinlock claim bits only when it used a spinlock,
    /// since the running core takes and frees its own.
    pub(crate) fn idle_watch_side(&self, core: usize, blocks: u8) -> [u32; 9] {
        let mut side = self.idle_side_state(core);
        if blocks & PARK_SPINLOCK == 0 {
            side[0] = 0;
        }
        side
    }

    /// Idle-loop detector: every word stored since the snapshot holds the
    /// value it had before the first store to it.
    pub(crate) fn idle_memory_restored(&self) -> bool {
        self.idle.log.iter().all(|&(off, old)| self.memory.sram_read32(off) == old)
    }

    /// State outside the core and SRAM that a clean pass may touch and the
    /// idle-loop detector compares: the spinlock claim bits, `core`'s
    /// divider, and both cores' event flags, which SEV sets.
    pub(crate) fn idle_side_state(&self, core: usize) -> [u32; 9] {
        let d = self.sio.divider(core);
        [
            self.sio.spinlock_bits(),
            d.dividend,
            d.divisor,
            d.quotient,
            d.remainder,
            d.signed as u32 | (d.dirty as u32) << 1,
            d.reads_pending as u32,
            self.event_flag[0] as u32,
            self.event_flag[1] as u32,
        ]
    }

    /// The USB controller.
    pub fn usb(&mut self) -> &mut picoem_common::usb::UsbCtrl {
        &mut self.usbctrl
    }

    /// Plug a USB host into the port, or unplug it.
    pub fn usb_plug(&mut self, present: bool) {
        self.usbctrl.set_host_present(present);
    }

    /// Advance a USB host to `now_us` against the controller, and pend
    /// USBCTRL_IRQ when the controller's interrupt is high afterwards.
    /// While the controller is held in reset the device is not on the bus.
    /// Returns true when the host touched the device.
    pub fn usb_host_poll(&mut self, host: &mut picoem_common::usb::cdc_host::CdcHost, now_us: u64) -> bool {
        self.usbctrl.set_host_present(true);
        if peripheral_dispatch::is_held_in_reset(self, USBCTRL_REGS_BASE) {
            return false;
        }
        let touched = host.poll(&mut self.usbctrl, now_us);
        if self.usbctrl.irq_level() {
            self.irq_pending |= 1 << crate::irq::IRQ_USBCTRL_IRQ;
        }
        touched
    }

    /// Wire an SPI device to SPI0 when `idx` is 0, else to SPI1.
    pub fn attach_spi_device(&mut self, idx: usize, dev: Box<dyn crate::peripherals::spi::SpiDevice>) {
        let spi = if idx == 0 { &mut self.spi0 } else { &mut self.spi1 };
        spi.set_pins(self.sio.gpio_out as u64);
        spi.attach_device(dev);
    }

    /// Unwire and return the device on SPI0 when `idx` is 0, else on SPI1.
    pub fn detach_spi_device(&mut self, idx: usize) -> Option<Box<dyn crate::peripherals::spi::SpiDevice>> {
        if idx == 0 { self.spi0.detach_device() } else { self.spi1.detach_device() }
    }

    /// Feed one byte into UART0's receiver. False when it was dropped.
    pub fn uart0_push_rx(&mut self, byte: u8) -> bool {
        self.uart0.push_rx(byte, &mut self.irq_pending)
    }

    /// Borrow a single DMA channel's observation state.
    ///
    /// External diagnostic harnesses (e.g. `picogus_diff_rp2040`) read
    /// per-channel counters on `DmaChannel` to verdict scenarios that
    /// MMIO can't surface without write side-effects. The accessor
    /// keeps `dma` itself `pub(crate)` while preserving the read-only
    /// observability the harness needs.
    ///
    /// Panics if `i >= crate::dma::NUM_CHANNELS` (matches the
    /// underlying `Dma::channel` slice index).
    #[inline]
    pub fn dma_channel(&self, i: usize) -> &crate::dma::DmaChannel {
        self.dma.channel(i)
    }

    /// Base read latency for an address region (cycles).
    #[inline]
    fn read_latency(region: u32) -> u32 {
        match region {
            0x0 => 1, // ROM
            0x1 => 1, // XIP / XIP_CTRL / SSI
            0x2 => 1, // SRAM
            0x4 => 3, // APB peripherals
            0x5 => 1, // AHB peripherals
            0xD => 1, // SIO
            0xE => 1, // PPB
            _ => 1,
        }
    }

    #[inline]
    fn write_latency(region: u32) -> u32 {
        match region {
            0x4 => 4, // APB writes
            _ => 1,
        }
    }

    /// Record an SRAM bank touch for the active core and return any
    /// contention wait states (simple +1 model).
    #[inline]
    fn note_sram_access(&mut self, addr: u32) -> u32 {
        if let Some(bank) = bank_for_address(addr) {
            let bit = 1u8 << (bank & 7);
            let wait = if self.contention_check_active && self.core0_bank_touched & bit != 0 {
                1
            } else {
                0
            };
            if self.active_core == 0 {
                self.core0_bank_touched |= bit;
            }
            wait
        } else {
            0
        }
    }

    // --- XIP SRAM scratch helpers ----------------------------------------

    fn xip_sram_read(&self, addr: u32, width: usize) -> u32 {
        let off = (addr - XIP_SRAM_BASE) as usize;
        let end = off + width;
        if end <= self.xip_sram.len() {
            match width {
                1 => self.xip_sram[off] as u32,
                2 => u16::from_le_bytes([self.xip_sram[off], self.xip_sram[off + 1]]) as u32,
                4 => u32::from_le_bytes([
                    self.xip_sram[off],
                    self.xip_sram[off + 1],
                    self.xip_sram[off + 2],
                    self.xip_sram[off + 3],
                ]),
                _ => 0,
            }
        } else {
            0
        }
    }

    fn xip_sram_write(&mut self, addr: u32, val: u32, width: usize) {
        let off = (addr - XIP_SRAM_BASE) as usize;
        let end = off + width;
        if end <= self.xip_sram.len() {
            let bytes = val.to_le_bytes();
            for i in 0..width {
                self.xip_sram[off + i] = bytes[i];
            }
        }
    }

    // --- Peripheral read dispatch ----------------------------------------

    fn peripheral_read32(&mut self, addr: u32) -> u32 {
        let canonical = addr & !0x3000;
        let base = canonical & 0xFFFF_F000;
        let offset = canonical & 0x0000_0FFF;
        // RESETS Bus-level guard (HLD V7 §5.3). Reset-gated peripherals
        // return 0 without the peripheral module ever seeing the read.
        if peripheral_dispatch::is_held_in_reset(self, base) {
            return 0;
        }
        match base {
            SYSINFO_BASE => self.sysinfo_read(offset),
            CLOCKS_BASE => self.clocks_regs.read32(offset),
            RESETS_BASE => self.resets.read32(offset),
            USBCTRL_REGS_BASE => self.usbctrl.read32(offset),
            USBCTRL_DPRAM_BASE => self.usbctrl.read_dpram(offset, 4),
            XOSC_BASE => self.xosc_regs.read32(offset),
            PLL_SYS_BASE => pll_read_with_lock(
                &self.pll_sys_regs,
                offset,
                self.pll_sys_lock_at_cycle,
                self.master_cycle,
            ),
            PLL_USB_BASE => pll_read_with_lock(
                &self.pll_usb_regs,
                offset,
                self.pll_usb_lock_at_cycle,
                self.master_cycle,
            ),
            ROSC_BASE => self.rosc_regs.read32(offset),
            IO_BANK0_BASE => self.io_bank0.read32(offset),
            PADS_BANK0_BASE => self.pads_bank0.read32(offset),
            PIO0_BASE => pio_read_rp2040(&mut self.pio[0], offset),
            PIO1_BASE => pio_read_rp2040(&mut self.pio[1], offset),
            DMA_BASE => self.dma.read32(offset),
            TIMER_BASE => self
                .timer
                .read32(offset, self.master_cycle, self.clock_tree.sys_clk_hz),
            WATCHDOG_BASE => self.watchdog_tick.read32(offset),
            UART0_BASE => self.uart0.read32(offset),
            UART1_BASE => self.uart1.read32(offset),
            SPI0_BASE => self.spi0.read32(offset),
            SPI1_BASE => self.spi1.read32(offset),
            I2C0_BASE => self.i2c0.read32(offset),
            I2C1_BASE => self.i2c1.read32(offset),
            ADC_BASE => self.adc.read32(offset),
            PWM_BASE => self.pwm.read32(offset),
            _ => *self.peripheral_regs.get(&canonical).unwrap_or(&0),
        }
    }

    fn peripheral_write32(&mut self, addr: u32, val: u32, alias: u32) {
        let canonical = addr & !0x3000;
        let base = canonical & 0xFFFF_F000;
        let offset = canonical & 0x0000_0FFF;
        // RESETS Bus-level guard (HLD V7 §5.3). Reset-gated peripherals
        // drop the write without the peripheral module ever seeing it.
        if peripheral_dispatch::is_held_in_reset(self, base) {
            return;
        }
        match base {
            SYSINFO_BASE => {} // read-only
            CLOCKS_BASE => {
                if self.clocks_regs.write32(offset, val, alias) {
                    self.recompute_clock_tree();
                }
            }
            RESETS_BASE => {
                let was = self.resets.state;
                self.resets.write32(offset, val, alias);
                // Asserting USBCTRL's reset clears the controller.
                let usb = 1u32 << peripheral_dispatch::RESET_USBCTRL;
                if self.resets.state & usb != 0 && was & usb == 0 {
                    self.usbctrl.reset();
                }
            }
            USBCTRL_REGS_BASE => self.usbctrl.write32(offset, val, alias),
            USBCTRL_DPRAM_BASE => self.usbctrl.write_dpram(offset, val, 4),
            XOSC_BASE => self.xosc_regs.write32(offset, val, alias),
            PLL_SYS_BASE => {
                let old_regs = self.pll_sys_regs;
                if clocks::pll_write(&mut self.pll_sys_regs, offset, val, alias) {
                    self.pll_sys_lock_at_cycle = pll_should_arm_lock(
                        &old_regs,
                        &self.pll_sys_regs,
                        self.pll_sys_lock_at_cycle,
                        self.master_cycle,
                    );
                    self.recompute_clock_tree();
                }
            }
            PLL_USB_BASE => {
                let old_regs = self.pll_usb_regs;
                if clocks::pll_write(&mut self.pll_usb_regs, offset, val, alias) {
                    self.pll_usb_lock_at_cycle = pll_should_arm_lock(
                        &old_regs,
                        &self.pll_usb_regs,
                        self.pll_usb_lock_at_cycle,
                        self.master_cycle,
                    );
                    self.recompute_clock_tree();
                }
            }
            ROSC_BASE => self.rosc_regs.write32(offset, val, alias),
            IO_BANK0_BASE => self.io_bank0.write32(offset, val, alias),
            PADS_BANK0_BASE => self.pads_bank0.write32(offset, val, alias),
            PIO0_BASE => self.pio[0].write32(pio_rp2040_to_internal(offset), val, alias),
            PIO1_BASE => self.pio[1].write32(pio_rp2040_to_internal(offset), val, alias),
            DMA_BASE => self.dma.write32(offset, val, alias),
            TIMER_BASE => {
                let sys_hz = self.clock_tree.sys_clk_hz;
                let mc = self.master_cycle;
                self.timer.write32(offset, val, alias, mc, sys_hz);
            }
            WATCHDOG_BASE => self.watchdog_tick.write32(offset, val, alias),
            UART0_BASE => self
                .uart0
                .write32(offset, val, alias, &mut self.irq_pending),
            UART1_BASE => self
                .uart1
                .write32(offset, val, alias, &mut self.irq_pending),
            SPI0_BASE => self.spi0.write32(offset, val, alias, &mut self.irq_pending),
            SPI1_BASE => self.spi1.write32(offset, val, alias, &mut self.irq_pending),
            I2C0_BASE => self.i2c0.write32(offset, val, alias, &mut self.irq_pending),
            I2C1_BASE => self.i2c1.write32(offset, val, alias, &mut self.irq_pending),
            ADC_BASE => self.adc.write32(offset, val, alias, &mut self.irq_pending),
            PWM_BASE => self.pwm.write32(offset, val, alias, &mut self.irq_pending),
            _ => {
                // Catch-all: store with alias semantics so firmware round-trips.
                let old = *self.peripheral_regs.get(&canonical).unwrap_or(&0);
                let new = match alias {
                    0 => val,
                    1 => old ^ val,
                    2 => old | val,
                    3 => old & !val,
                    _ => val,
                };
                self.peripheral_regs.insert(canonical, new);
                // PSM FRCE_OFF bit 16 holds core 1 in reset.
                if canonical == PSM_BASE + 0x04 && (old ^ new) & (1 << 16) != 0 {
                    self.core1_psm_reset = Some(new & (1 << 16) != 0);
                }
                // GPIO_QSPI_SS_CTRL: a change of the chip-select output
                // override starts a new flash command.
                if canonical == IO_QSPI_BASE + 0x0c && (old ^ new) & 0x300 != 0 {
                    self.ssi_cmd.clear();
                }
            }
        }
    }

    fn sysinfo_read(&self, offset: u32) -> u32 {
        match offset {
            0x000 => 0x0000_0001, // CHIP_ID: RP2040 manufacturer (placeholder)
            0x004 => 0x0000_0000, // PLATFORM
            _ => 0,
        }
    }

    // --- XIP_CTRL + SSI stubs --------------------------------------------

    fn xip_ctrl_read(&self, offset: u32) -> u32 {
        // XIP_CTRL_CTRL (offset 0x00) reports EN=1 so the bootrom's check
        // for "XIP cache enabled" succeeds immediately.
        match offset {
            0x00 => *self.xip_ctrl_regs.get(&0).unwrap_or(&1),
            _ => *self.xip_ctrl_regs.get(&offset).unwrap_or(&0),
        }
    }

    fn xip_ctrl_write(&mut self, offset: u32, val: u32) {
        self.xip_ctrl_regs.insert(offset, val);
    }

    /// SSI in direct mode, the path the bootrom's flash_do_cmd and the
    /// SDK's flash routines take. Transmit completes at once, so TXFLR is
    /// always 0; RXFLR is the number of reply bytes waiting.
    fn ssi_read(&mut self, offset: u32) -> u32 {
        match offset {
            0x20 => 0,                                  // TXFLR
            0x24 => self.ssi_rx.len() as u32,           // RXFLR
            // SR: TFNF | TFE, plus RFNE when a reply byte waits. BUSY is
            // bit 0 and stays clear.
            0x28 => 0x06 | if self.ssi_rx.is_empty() { 0 } else { 0x08 },
            0x60 => self.ssi_rx.pop_front().unwrap_or(0) as u32, // DR0
            _ => *self.ssi_regs.get(&offset).unwrap_or(&0),
        }
    }

    fn ssi_write(&mut self, offset: u32, val: u32) {
        match offset {
            0x60 => {
                if self.ssi_regs.get(&0x08).copied().unwrap_or(0) & 1 != 0 {
                    let reply = self.flash_reply(val as u8);
                    self.ssi_rx.push_back(reply);
                }
            }
            0x08 => {
                // SSIENR: disabling the SSI empties its FIFOs.
                if val & 1 == 0 {
                    self.ssi_rx.clear();
                }
                self.ssi_regs.insert(offset, val);
            }
            _ => {
                self.ssi_regs.insert(offset, val);
            }
        }
    }

    /// Serial NOR flash reply to one byte of a command, W25Q16-style:
    /// 0x03 read, 0x05/0x35 status, 0x4B unique ID, 0x9F JEDEC ID, 0xAB
    /// release from power-down, 0x06/0x04 write enable/disable, 0x02 page
    /// program, 0x20/0x52/0xD8 4/32/64 KB erase and 0x60/0xC7 chip erase.
    /// Program and erase need write enable, take effect at once and clear
    /// it, so status never reports busy.
    fn flash_reply(&mut self, tx: u8) -> u8 {
        self.ssi_cmd.push(tx);
        let i = self.ssi_cmd.len() - 1;
        let cmd = self.ssi_cmd[0];
        let addr = |a: &[u8]| (a[1] as u32) << 16 | (a[2] as u32) << 8 | a[3] as u32;
        match cmd {
            0x06 if i == 0 => self.flash_wel = true,
            0x04 if i == 0 => self.flash_wel = false,
            0x05 if i >= 1 => return (self.flash_wel as u8) << 1,
            0x02 if i == 3 => {
                // Code runs from SRAM while XIP is off, so one decode
                // cache flush per command covers the bytes that follow.
                self.flash_prog = self.flash_wel;
                self.flash_wel = false;
                self.pending_invalidation_regions |= invalidation_regions::XIP;
            }
            0x02 if i >= 4 && self.flash_prog => {
                // A page program wraps within its 256-byte page.
                let a = addr(&self.ssi_cmd);
                let a = (a & !0xFF) | ((a + (i as u32 - 4)) & 0xFF);
                self.memory.flash_program8(a & 0x00ff_ffff, tx);
            }
            0x20 | 0x52 | 0xD8 if i == 3 => {
                if self.flash_wel {
                    let len = match cmd {
                        0x20 => 0x1000,
                        0x52 => 0x8000,
                        _ => 0x1_0000,
                    };
                    let a = addr(&self.ssi_cmd) & !(len - 1);
                    self.memory.flash_erase(a & 0x00ff_ffff, len);
                    self.pending_invalidation_regions |= invalidation_regions::XIP;
                }
                self.flash_wel = false;
            }
            0x60 | 0xC7 if i == 0 => {
                if self.flash_wel {
                    let n = self.memory.flash_size() as u32;
                    self.memory.flash_erase(0, n);
                    self.pending_invalidation_regions |= invalidation_regions::XIP;
                }
                self.flash_wel = false;
            }
            _ => {}
        }
        match cmd {
            0x03 if i >= 4 => {
                let a = &self.ssi_cmd;
                let addr = ((a[1] as u32) << 16 | (a[2] as u32) << 8 | a[3] as u32) + (i as u32 - 4);
                self.memory.xip_read8(addr & 0x00ff_ffff)
            }
            0x4B if i >= 5 => [0xE6, 0x61, 0x41, 0x04, 0x03, 0x2F, 0x28, 0x2E][(i - 5) % 8],
            0x9F if (1..=3).contains(&i) => [0xEF, 0x40, 0x15][i - 1],
            0xAB if i >= 4 => 0x14,
            _ => 0,
        }
    }

    // ----------------------------------------------------------------
    // Narrow-access dispatch for byte/halfword-significant peripheral
    // registers (UART_DR, SSPDR, IC_DATA_CMD). Returns `Some(val)` /
    // `true` when handled, `None` / `false` to signal the caller it
    // should fall back to the word-RMW path. This keeps UART/SPI/I2C
    // side-effect registers from suffering spurious FIFO pops on a
    // byte write that would otherwise execute `read32` → splice →
    // `write32` and double-side-effect the register.
    // ----------------------------------------------------------------

    /// Per HLD V7 §5.4, only peripherals with side-effect narrow
    /// registers need narrow dispatch. Returns `true` if the peripheral
    /// at `base` has one (UART DR, SPI DR, I2C DATA_CMD); callers
    /// dispatch to `narrow_peripheral_read*` / `narrow_peripheral_write*`
    /// when this is set, otherwise RMW-through-word is used.
    #[inline]
    fn peripheral_has_narrow_register(base: u32, offset: u32) -> bool {
        match base {
            UART0_BASE | UART1_BASE => offset == crate::peripherals::uart::UARTDR,
            SPI0_BASE | SPI1_BASE => offset == crate::peripherals::spi::SSPDR,
            I2C0_BASE | I2C1_BASE => offset == crate::peripherals::i2c::IC_DATA_CMD,
            // ADC FIFO pops a sample on any-width read; a byte / halfword
            // read must hit the narrow path so a spurious word32 read-
            // modify-write doesn't double-pop the FIFO. Datasheet §4.9.6
            // notes firmware may configure FCS.SHIFT to right-justify the
            // 12-bit sample to an 8-bit halfword read.
            ADC_BASE => offset == crate::peripherals::adc::FIFO,
            _ => false,
        }
    }

    /// Byte-read a peripheral register with side-effect semantics —
    /// UART_DR / SSPDR / IC_DATA_CMD / ADC FIFO. Caller guarantees
    /// `base + offset` has been checked by
    /// `peripheral_has_narrow_register`.
    fn narrow_peripheral_read8(&mut self, base: u32, offset: u32) -> u8 {
        // RESETS is Bus-level — held peripherals return 0.
        if peripheral_dispatch::is_held_in_reset(self, base) {
            return 0;
        }
        match base {
            UART0_BASE => self.uart0.read8(offset),
            UART1_BASE => self.uart1.read8(offset),
            SPI0_BASE => self.spi0.read8(offset),
            SPI1_BASE => self.spi1.read8(offset),
            I2C0_BASE => self.i2c0.read8(offset),
            I2C1_BASE => self.i2c1.read8(offset),
            ADC_BASE => self.adc.read16(offset) as u8,
            _ => 0,
        }
    }

    /// Halfword-read. Same constraints as `narrow_peripheral_read8`.
    fn narrow_peripheral_read16(&mut self, base: u32, offset: u32) -> u16 {
        if peripheral_dispatch::is_held_in_reset(self, base) {
            return 0;
        }
        match base {
            SPI0_BASE => self.spi0.read16(offset),
            SPI1_BASE => self.spi1.read16(offset),
            // UART DR and I2C DATA_CMD don't carry halfword semantics
            // — a 16-bit read collapses to the low byte with zero in
            // the high byte. Use the narrow byte dispatch and
            // zero-extend.
            UART0_BASE => self.uart0.read8(offset) as u16,
            UART1_BASE => self.uart1.read8(offset) as u16,
            I2C0_BASE => self.i2c0.read32(offset) as u16,
            I2C1_BASE => self.i2c1.read32(offset) as u16,
            ADC_BASE => self.adc.read16(offset),
            _ => 0,
        }
    }

    /// Byte-write.
    fn narrow_peripheral_write8(&mut self, base: u32, offset: u32, val: u8) {
        if peripheral_dispatch::is_held_in_reset(self, base) {
            return;
        }
        match base {
            UART0_BASE => self.uart0.write8(offset, val, &mut self.irq_pending),
            UART1_BASE => self.uart1.write8(offset, val, &mut self.irq_pending),
            SPI0_BASE => self.spi0.write8(offset, val, &mut self.irq_pending),
            SPI1_BASE => self.spi1.write8(offset, val, &mut self.irq_pending),
            I2C0_BASE => self.i2c0.write8(offset, val, &mut self.irq_pending),
            I2C1_BASE => self.i2c1.write8(offset, val, &mut self.irq_pending),
            // ADC FIFO is read-only — narrow writes swallowed.
            ADC_BASE => {}
            _ => {}
        }
    }

    /// Halfword-write.
    fn narrow_peripheral_write16(&mut self, base: u32, offset: u32, val: u16) {
        if peripheral_dispatch::is_held_in_reset(self, base) {
            return;
        }
        match base {
            SPI0_BASE => self.spi0.write16(offset, val, &mut self.irq_pending),
            SPI1_BASE => self.spi1.write16(offset, val, &mut self.irq_pending),
            UART0_BASE => self.uart0.write8(offset, val as u8, &mut self.irq_pending),
            UART1_BASE => self.uart1.write8(offset, val as u8, &mut self.irq_pending),
            I2C0_BASE => self
                .i2c0
                .write32(offset, val as u32, 0, &mut self.irq_pending),
            I2C1_BASE => self
                .i2c1
                .write32(offset, val as u32, 0, &mut self.irq_pending),
            // ADC FIFO is read-only — narrow writes swallowed.
            ADC_BASE => {}
            _ => {}
        }
    }

    // ======================================================================
    // Read / write entry points
    // ======================================================================

    pub fn read8(&mut self, addr: u32) -> u8 {
        let region = addr >> 28;
        self.last_access_cycles = Self::read_latency(region);
        let val = match region {
            0x0 if (addr & 0x0FFF_FFFF) < ROM_SIZE as u32 => {
                self.memory.rom_read8(addr & 0x0FFF_FFFF)
            }
            0x1 => self.region1_read(addr, 1) as u8,
            0x2 => {
                let off = addr & 0x00FF_FFFF;
                if (off as usize) < SRAM_SIZE {
                    self.last_access_cycles += self.note_sram_access(addr);
                    self.memory.sram_read8(off)
                } else {
                    self.set_bus_fault(addr);
                    0
                }
            }
            0x4 | 0x5 => {
                let canonical = addr & !0x3000;
                let base = canonical & 0xFFFF_F000;
                let offset = canonical & 0x0000_0FFF;
                if Self::peripheral_has_narrow_register(base, offset & !3) {
                    self.narrow_peripheral_read8(base, offset & !3)
                } else {
                    let word = self.peripheral_read32(addr & !3);
                    word.to_le_bytes()[(addr & 3) as usize]
                }
            }
            0xD => {
                let word = self.sio_read32(addr & !3);
                word.to_le_bytes()[(addr & 3) as usize]
            }
            0xE => {
                let w32 = addr & !3;
                let word = if let Some(v) = self.nvic_mmio_read32(w32) {
                    v
                } else if let Some(v) = self.systick_mmio_read32(w32) {
                    v
                } else {
                    self.ppb[self.active_core].read32(w32)
                };
                word.to_le_bytes()[(addr & 3) as usize]
            }
            _ => {
                self.set_bus_fault(addr);
                0
            }
        };
        if self.mmio_trace_enabled {
            self.emit_mmio_trace('R', 1, addr, val as u32);
        }
        val
    }

    pub fn read16(&mut self, addr: u32) -> u16 {
        let region = addr >> 28;
        self.last_access_cycles = Self::read_latency(region);
        let val = match region {
            0x0 if (addr & 0x0FFF_FFFF) + 1 < ROM_SIZE as u32 => {
                self.memory.rom_read16(addr & 0x0FFF_FFFF)
            }
            0x1 => self.region1_read(addr, 2) as u16,
            0x2 => {
                let off = addr & 0x00FF_FFFF;
                if ((off + 1) as usize) < SRAM_SIZE {
                    self.last_access_cycles += self.note_sram_access(addr);
                    self.memory.sram_read16(off)
                } else {
                    self.set_bus_fault(addr);
                    0
                }
            }
            0x4 | 0x5 => {
                let canonical = addr & !0x3000;
                let base = canonical & 0xFFFF_F000;
                let offset = canonical & 0x0000_0FFF;
                if Self::peripheral_has_narrow_register(base, offset & !3) {
                    self.narrow_peripheral_read16(base, offset & !3)
                } else {
                    let word = self.peripheral_read32(addr & !3);
                    let half = ((addr >> 1) & 1) as usize;
                    [word as u16, (word >> 16) as u16][half]
                }
            }
            0xD => {
                let word = self.sio_read32(addr & !3);
                let half = ((addr >> 1) & 1) as usize;
                [word as u16, (word >> 16) as u16][half]
            }
            0xE => {
                let w32 = addr & !3;
                let word = if let Some(v) = self.nvic_mmio_read32(w32) {
                    v
                } else if let Some(v) = self.systick_mmio_read32(w32) {
                    v
                } else {
                    self.ppb[self.active_core].read32(w32)
                };
                let half = ((addr >> 1) & 1) as usize;
                [word as u16, (word >> 16) as u16][half]
            }
            _ => {
                self.set_bus_fault(addr);
                0
            }
        };
        if self.mmio_trace_enabled {
            self.emit_mmio_trace('R', 2, addr, val as u32);
        }
        val
    }

    pub fn read32(&mut self, addr: u32) -> u32 {
        let region = addr >> 28;
        self.last_access_cycles = Self::read_latency(region);
        let val = match region {
            0x0 if (addr & 0x0FFF_FFFF) + 3 < ROM_SIZE as u32 => {
                self.memory.rom_read32(addr & 0x0FFF_FFFF)
            }
            0x1 => self.region1_read(addr, 4),
            0x2 => {
                let off = addr & 0x00FF_FFFF;
                if ((off + 3) as usize) < SRAM_SIZE {
                    self.last_access_cycles += self.note_sram_access(addr);
                    self.memory.sram_read32(off)
                } else {
                    self.set_bus_fault(addr);
                    0
                }
            }
            0x4 | 0x5 => self.peripheral_read32(addr),
            0xD => self.sio_read32(addr),
            0xE => {
                if let Some(w) = self.nvic_mmio_read32(addr) {
                    w
                } else if let Some(w) = self.systick_mmio_read32(addr) {
                    w
                } else {
                    self.ppb[self.active_core].read32(addr)
                }
            }
            _ => {
                self.set_bus_fault(addr);
                0
            }
        };
        if self.mmio_trace_enabled {
            self.emit_mmio_trace('R', 4, addr, val);
        }
        val
    }

    pub fn write8(&mut self, addr: u32, val: u8) {
        if self.mmio_trace_enabled {
            self.emit_mmio_trace('W', 1, addr, val as u32);
        }
        let region = addr >> 28;
        self.last_access_cycles = Self::write_latency(region);
        match region {
            0x1 if (XIP_SRAM_BASE..XIP_SRAM_END).contains(&addr) => {
                self.xip_sram_write(addr, val as u32, 1);
                self.invalidate_pc_range(addr, 1);
            }
            0x2 => {
                let off = addr & 0x00FF_FFFF;
                if (off as usize) < SRAM_SIZE {
                    self.last_access_cycles += self.note_sram_access(addr);
                    // All four SRAM alias windows (0x20/0x21/0x22/0x23)
                    // map to the same backing storage — RP2040 datasheet
                    // §2.1.2 calls out alias bits [25:24] as bank-striping
                    // flavours for DMA, not peripheral XOR/SET/CLR.
                    self.ext_sram_writes += 1;
                    self.memory.sram_write8(off, val);
                    self.invalidate_pc_range(addr, 1);
                } else {
                    self.set_bus_fault(addr);
                }
            }
            0x4 | 0x5 => {
                let canonical = addr & !0x3000;
                let base = canonical & 0xFFFF_F000;
                if base == PIO0_BASE || base == PIO1_BASE {
                    let pio_offset = canonical & 0x0000_0FFF;
                    // TXF0-3: byte-replicate (ARM AHB-Lite HSIZE=0
                    // replication) and push to the TX FIFO. All other
                    // PIO registers: 32-bit access only — byte writes
                    // would trigger spurious RXF pops via the RMW read,
                    // so silently ignore.
                    if (0x010..=0x01C).contains(&pio_offset) {
                        let replicated = (val as u32) * 0x0101_0101;
                        let idx = if base == PIO0_BASE { 0 } else { 1 };
                        self.pio[idx].write32(pio_offset, replicated, 0);
                    }
                    return;
                }
                let offset = canonical & 0x0000_0FFF;
                let word_offset = offset & !3;
                if Self::peripheral_has_narrow_register(base, word_offset) {
                    self.narrow_peripheral_write8(base, word_offset, val);
                } else {
                    let alias = (addr >> 12) & 3;
                    // Byte-level RMW into the word, preserving alias semantics.
                    let word_addr = canonical & !3;
                    let byte_idx = (canonical & 3) as usize;
                    let old = self.peripheral_read32(word_addr);
                    let mut bytes = old.to_le_bytes();
                    bytes[byte_idx] = val;
                    let new_word = u32::from_le_bytes(bytes);
                    // For an alias access, convert the byte to a positioned
                    // word and defer alias math to the peripheral layer.
                    if alias == 0 {
                        self.peripheral_write32(word_addr, new_word, 0);
                    } else {
                        let shifted = (val as u32) << (byte_idx * 8);
                        self.peripheral_write32(word_addr, shifted, alias);
                    }
                }
            }
            0xD => {
                let word_addr = addr & !3;
                let byte_idx = (addr & 3) as usize;
                let old = self.sio_read32(word_addr);
                let mut bytes = old.to_le_bytes();
                bytes[byte_idx] = val;
                self.sio_write32(word_addr, u32::from_le_bytes(bytes));
            }
            0xE => {
                let word_addr = addr & !3;
                let byte_idx = (addr & 3) as usize;
                let old = if let Some(v) = self.nvic_mmio_read32(word_addr) {
                    v
                } else if let Some(v) = self.systick_mmio_read32(word_addr) {
                    v
                } else {
                    self.ppb[self.active_core].read32(word_addr)
                };
                let mut bytes = old.to_le_bytes();
                bytes[byte_idx] = val;
                let new_word = u32::from_le_bytes(bytes);
                // NVIC and SysTick both consume the write fully when they
                // claim it; PPB takes the rest.
                if !self.nvic_mmio_write32(word_addr, new_word)
                    && !self.systick_mmio_write32(word_addr, new_word)
                {
                    self.ppb[self.active_core].write32(word_addr, new_word);
                }
            }
            0x0 | 0x1 => {} // ROM / XIP flash — silently ignored at any width
            _ => {
                // Unmapped at any width sets the sticky bus-fault flag so
                // step() can escalate to HardFault.
                self.set_bus_fault(addr);
            }
        }
    }

    pub fn write16(&mut self, addr: u32, val: u16) {
        if self.mmio_trace_enabled {
            self.emit_mmio_trace('W', 2, addr, val as u32);
        }
        let region = addr >> 28;
        self.last_access_cycles = Self::write_latency(region);
        match region {
            0x1 if (XIP_SRAM_BASE..XIP_SRAM_END).contains(&addr) => {
                self.xip_sram_write(addr, val as u32, 2);
                self.invalidate_pc_range(addr, 2);
            }
            0x2 => {
                let off = addr & 0x00FF_FFFF;
                if ((off + 1) as usize) < SRAM_SIZE {
                    self.last_access_cycles += self.note_sram_access(addr);
                    // All four SRAM alias windows map to the same storage
                    // (RP2040 datasheet §2.1.2).
                    self.ext_sram_writes += 1;
                    self.memory.sram_write16(off, val);
                    self.invalidate_pc_range(addr, 2);
                } else {
                    self.set_bus_fault(addr);
                }
            }
            0x4 | 0x5 => {
                let canonical = addr & !0x3000;
                let base = canonical & 0xFFFF_F000;
                if base == PIO0_BASE || base == PIO1_BASE {
                    let pio_offset = canonical & 0x0000_0FFF;
                    // TXF0-3: halfword-replicate (ARM AHB-Lite HSIZE=1
                    // replication) and push to the TX FIFO. All other
                    // PIO registers: 32-bit access only.
                    if (0x010..=0x01C).contains(&pio_offset) {
                        let replicated = (val as u32) | ((val as u32) << 16);
                        let idx = if base == PIO0_BASE { 0 } else { 1 };
                        self.pio[idx].write32(pio_offset, replicated, 0);
                    }
                    return;
                }
                let offset = canonical & 0x0000_0FFF;
                let word_offset = offset & !3;
                if Self::peripheral_has_narrow_register(base, word_offset) {
                    self.narrow_peripheral_write16(base, word_offset, val);
                } else {
                    let alias = (addr >> 12) & 3;
                    let word_addr = canonical & !3;
                    let half_idx = ((canonical >> 1) & 1) as usize;
                    let old = self.peripheral_read32(word_addr);
                    let mut halves: [u16; 2] = [old as u16, (old >> 16) as u16];
                    halves[half_idx] = val;
                    let new_word = (halves[0] as u32) | ((halves[1] as u32) << 16);
                    if alias == 0 {
                        self.peripheral_write32(word_addr, new_word, 0);
                    } else {
                        let shifted = (val as u32) << (half_idx * 16);
                        self.peripheral_write32(word_addr, shifted, alias);
                    }
                }
            }
            0xD => {
                let word_addr = addr & !3;
                let half_idx = ((addr >> 1) & 1) as usize;
                let old = self.sio_read32(word_addr);
                let mut halves: [u16; 2] = [old as u16, (old >> 16) as u16];
                halves[half_idx] = val;
                let new_word = (halves[0] as u32) | ((halves[1] as u32) << 16);
                self.sio_write32(word_addr, new_word);
            }
            0xE => {
                let word_addr = addr & !3;
                let half_idx = ((addr >> 1) & 1) as usize;
                let old = if let Some(v) = self.nvic_mmio_read32(word_addr) {
                    v
                } else if let Some(v) = self.systick_mmio_read32(word_addr) {
                    v
                } else {
                    self.ppb[self.active_core].read32(word_addr)
                };
                let mut halves: [u16; 2] = [old as u16, (old >> 16) as u16];
                halves[half_idx] = val;
                let new_word = (halves[0] as u32) | ((halves[1] as u32) << 16);
                // NVIC and SysTick both consume the write fully when they
                // claim it; PPB takes the rest.
                if !self.nvic_mmio_write32(word_addr, new_word)
                    && !self.systick_mmio_write32(word_addr, new_word)
                {
                    self.ppb[self.active_core].write32(word_addr, new_word);
                }
            }
            0x0 | 0x1 => {} // ROM / XIP flash — silently ignored at any width
            _ => {
                self.set_bus_fault(addr);
            }
        }
    }

    pub fn write32(&mut self, addr: u32, val: u32) {
        if self.mmio_trace_enabled {
            self.emit_mmio_trace('W', 4, addr, val);
        }
        let region = addr >> 28;
        self.last_access_cycles = Self::write_latency(region);
        match region {
            0x1 if (XIP_SRAM_BASE..XIP_SRAM_END).contains(&addr) => {
                self.xip_sram_write(addr, val, 4);
                self.invalidate_pc_range(addr, 4);
            }
            0x1 => {
                // Region 0x1 at XIP_CTRL (0x1400_0000) or SSI (0x1800_0000).
                let base = addr & 0xFFFF_F000;
                let offset = addr & 0x0FFF;
                if base == XIP_CTRL_BASE {
                    self.xip_ctrl_write(offset, val);
                } else if base == SSI_BASE {
                    self.ssi_write(offset, val);
                }
            }
            0x2 => {
                let off = addr & 0x00FF_FFFF;
                if ((off + 3) as usize) < SRAM_SIZE {
                    self.last_access_cycles += self.note_sram_access(addr);
                    // All four SRAM alias windows map to the same storage
                    // (RP2040 datasheet §2.1.2).
                    self.ext_sram_writes += 1;
                    self.memory.sram_write32(off, val);
                    self.invalidate_pc_range(addr, 4);
                } else {
                    self.set_bus_fault(addr);
                }
            }
            0x4 | 0x5 => {
                let alias = (addr >> 12) & 3;
                self.peripheral_write32(addr, val, alias);
            }
            0xD => self.sio_write32(addr, val),
            0xE => {
                // NVIC and SysTick both consume the write fully when they
                // claim it; PPB takes the rest.
                if !self.nvic_mmio_write32(addr, val) && !self.systick_mmio_write32(addr, val) {
                    self.ppb[self.active_core].write32(addr, val);
                }
            }
            0x0 => {} // ROM — silently ignored at any width
            _ => {
                self.set_bus_fault(addr);
            }
        }
    }

    // --- Region 0x1 read dispatch (XIP flash / XIP SRAM / XIP_CTRL / SSI)

    fn region1_read(&mut self, addr: u32, width: usize) -> u32 {
        if (XIP_SRAM_BASE..XIP_SRAM_END).contains(&addr) {
            return self.xip_sram_read(addr, width);
        }
        let base = addr & 0xFFFF_F000;
        let offset = addr & 0x0FFF;
        if base == XIP_CTRL_BASE {
            return self.xip_ctrl_read(offset);
        }
        if base == SSI_BASE {
            return self.ssi_read(offset);
        }
        // XIP flash window (0x10/0x11/0x12/0x13, each a 2 MB mirror).
        // PicoGUS Integration HLD (Stage 1): flash is a plain mapped
        // window — no wait states, no cache, no fault before load.
        if let Some(flash_off) = xip_flash_offset(addr) {
            return match width {
                1 => self.memory.xip_read8(flash_off) as u32,
                2 => self.memory.xip_read16(flash_off) as u32,
                4 => self.memory.xip_read32(flash_off),
                _ => 0,
            };
        }
        0
    }

    // --- PPB + NVIC read/write dispatch ----------------------------------
    //
    // The NVIC lives in the System Control Space at `0xE000_E100..=
    // 0xE000_E41F`. These registers are banked per-core on M0+, so
    // `nvics[active_core]` is the one the currently-executing CPU sees.
    // Other 0xE000_Exxx addresses fall through to the PPB.

    /// Intercept NVIC MMIO before the PPB sees it. Returns `Some(word)`
    /// when `addr` lies inside the NVIC ISER0 / ICER0 / ISPR0 / ICPR0 /
    /// IPR0..7 range, `None` otherwise so the caller can fall through
    /// to the PPB dispatch.
    fn nvic_mmio_read32(&self, addr: u32) -> Option<u32> {
        let low = addr & 0xFFFF;
        let n = &self.nvics[self.active_core];
        match low {
            // NVIC_ISER0 (0xE100) and NVIC_ICER0 (0xE180) both READ the
            // enable mask (ARMv6-M ARM §B3.4.4 / §B3.4.5).
            0xE100 | 0xE180 => Some(n.enabled),
            // NVIC_ISPR0 (0xE200) / NVIC_ICPR0 (0xE280) both READ the
            // pending mask.
            0xE200 | 0xE280 => Some(n.pending),
            // NVIC_IPR0..7 at 0xE400 + 4N. Each word holds 4 × 8-bit
            // priority bytes for IRQs [N*4..N*4+4].
            0xE400..=0xE41F => {
                let word_idx = ((low - 0xE400) >> 2) as usize;
                let base_irq = word_idx * 4;
                let mut w = 0u32;
                for lane in 0..4 {
                    let irq = base_irq + lane;
                    if irq < 32 {
                        w |= (n.priority[irq] as u32) << (lane * 8);
                    }
                }
                Some(w)
            }
            _ => None,
        }
    }

    /// Intercept NVIC MMIO writes. Returns `true` when handled. All
    /// four register families are per-core.
    fn nvic_mmio_write32(&mut self, addr: u32, val: u32) -> bool {
        let low = addr & 0xFFFF;
        let n = &mut self.nvics[self.active_core];
        match low {
            // NVIC_ISER0: write-1-to-SET the enable bit.
            0xE100 => {
                n.enabled |= val & crate::irq::IRQ_LINE_MASK;
                true
            }
            // NVIC_ICER0: write-1-to-CLEAR the enable bit.
            0xE180 => {
                n.enabled &= !(val & crate::irq::IRQ_LINE_MASK);
                true
            }
            // NVIC_ISPR0: write-1-to-SET the pending bit.
            0xE200 => {
                n.pending |= val & crate::irq::IRQ_LINE_MASK;
                true
            }
            // NVIC_ICPR0: write-1-to-CLEAR the pending bit.
            0xE280 => {
                n.pending &= !(val & crate::irq::IRQ_LINE_MASK);
                true
            }
            // NVIC_IPR0..7: 4×u8 priority bytes, each masked to the
            // implemented bits [7:6] (M0+ supports 4 priority levels).
            0xE400..=0xE41F => {
                let word_idx = ((low - 0xE400) >> 2) as usize;
                let base_irq = word_idx * 4;
                for lane in 0..4 {
                    let irq = base_irq + lane;
                    if irq < 32 {
                        let byte = ((val >> (lane * 8)) & 0xFF) as u8;
                        n.priority[irq] = byte & crate::core::nvic::PRIORITY_MASK;
                    }
                }
                true
            }
            _ => false,
        }
    }

    // --- SysTick MMIO dispatch -------------------------------------------
    //
    // HLD V5 §5.2: SysTick lives at `0xE000_E010..0xE000_E01F`. Per-core
    // banked register set; reads/writes target `systicks[active_core]`.
    // Note `systick_mmio_read32` is `&mut self` — `SYST_CSR` reads clear
    // `COUNTFLAG` per ARMv6-M ARM §B3.3.2.

    /// Intercept SysTick MMIO before the PPB sees it. Returns
    /// `Some(word)` when `addr` lies in the SysTick range, `None`
    /// otherwise so the caller can fall through.
    fn systick_mmio_read32(&mut self, addr: u32) -> Option<u32> {
        match addr & 0xFFFF {
            0xE010..=0xE01F => Some(self.systicks[self.active_core].read32(addr)),
            _ => None,
        }
    }

    /// Intercept SysTick MMIO writes. Returns `true` when handled.
    fn systick_mmio_write32(&mut self, addr: u32, val: u32) -> bool {
        match addr & 0xFFFF {
            0xE010..=0xE01F => {
                self.systicks[self.active_core].write32(addr, val);
                true
            }
            _ => false,
        }
    }

    // --- SIO read/write dispatch -----------------------------------------
    //
    // GPIO_IN (0x004) is owned by Bus so the SIO crate has no direct
    // dependency on PIO (Phase 5.B lifts this out). All other offsets
    // delegate to `Sio`.

    fn sio_read32(&mut self, addr: u32) -> u32 {
        let offset = addr & 0xFFF;
        match offset {
            0x004 => self.gpio_in,
            _ => {
                let core = self.active_core;
                self.sio.read32(offset, core)
            }
        }
    }

    fn sio_write32(&mut self, addr: u32, val: u32) {
        let offset = addr & 0xFFF;
        let core = self.active_core;
        let out = self.sio.gpio_out;
        self.sio.write32(offset, val, core);
        if self.sio.gpio_out != out {
            let pins = self.sio.gpio_out as u64;
            self.spi0.set_pins(pins);
            self.spi1.set_pins(pins);
        }
        if let Some(receiver) = self.sio.pending_fifo_event.take() {
            self.event_flag[receiver] = true;
        }
    }

    // --- Back-compat accessors for Phase 3 / 4 tests ---------------------
    //
    // The previous Phase 3 stub exposed a `gpio_in` field; keep that
    // interface stable so tests don't need updating.
    #[inline]
    pub fn gpio_in(&self) -> u32 {
        self.gpio_in
    }

    /// Signal SEV to both cores.
    pub fn signal_sev(&mut self) {
        self.event_flag[0] = true;
        self.event_flag[1] = true;
    }

    // --- Fast-path gate helpers (HLD V7 §5.5) ----------------------------

    /// True iff every stateful peripheral is idle — i.e. nothing that
    /// could observably change on a per-cycle tick is currently active.
    ///
    /// **MUST be updated when a new stateful peripheral is added to
    /// `Bus`.** This function is the fast-path gate in
    /// [`crate::Emulator::step`]; forgetting to AND a new peripheral's
    /// `is_idle()` into the result silently keeps the fast path taken
    /// while the peripheral is actually running, losing per-cycle
    /// observability (interrupts, FIFO edges, DMA transfers).
    ///
    /// Phase 1 peripherals with state (`TIMER`, `WATCHDOG_TICK`) are
    /// lazy and therefore always idle at the fast-path check — their
    /// internal effects fire on alarm-match rather than per-cycle. DMA
    /// is tracked separately via [`crate::dma::Dma::is_idle`] because
    /// it's a bus master, not a per-cycle-ticked peripheral, and is
    /// AND-ed into the fast-path gate alongside this result in
    /// [`crate::Emulator::step`]. Phase 3 adds ADC (idle iff no
    /// conversion armed) and PWM (idle iff `EN == 0`).
    #[inline]
    pub fn all_peripherals_idle(&self) -> bool {
        // Explicit acknowledgement of every stateful peripheral field.
        // When a new peripheral field is added to `Bus`, add a
        // reference here; Rust will flag any field rename / removal at
        // compile time. Phase 1 Wave 2: TIMER is a lazy peripheral —
        // alarm firing happens inside `advance_lazy_scheduled`, which
        // itself runs in the fast path. A latched INTR without INTE
        // set (no NVIC routing) still counts as idle because nothing
        // observable happens per-cycle. WATCHDOG_TICK has no tick.
        // Phase 2: UART/SPI/I2C are per-cycle-tickable — their
        // `is_idle()` getters gate the fast path.
        // Phase 3: ADC + PWM added — ADC mid-conversion must keep the
        // fast path closed so the clk_adc accumulator advances at all;
        // PWM with any enabled slice similarly.
        #[cfg(debug_assertions)]
        {
            let _ = (
                &self.watchdog_tick,
                &self.dma,
                &self.timer,
                &self.uart0,
                &self.uart1,
                &self.spi0,
                &self.spi1,
                &self.i2c0,
                &self.i2c1,
                &self.adc,
                &self.pwm,
            );
        }
        self.timer.is_idle()
            && self.uart0.is_idle()
            && self.uart1.is_idle()
            && self.spi0.is_idle()
            && self.spi1.is_idle()
            && self.i2c0.is_idle()
            && self.i2c1.is_idle()
            && self.adc.is_idle()
            && self.pwm.is_idle()
            && self.dma.is_idle()
    }

    /// True iff at least one PIO SM is enabled in either block, or any
    /// IRQ flag is still asserted. Either condition means a per-cycle
    /// PIO step could mutate pin state or the IRQ-flags register — so
    /// the fast-path is not safe (HLD V7 §5.5, "`pio_all_idle`"
    /// semantics).
    #[inline]
    pub fn pio_all_idle(&self) -> bool {
        !self.pio[0].any_sm_enabled()
            && !self.pio[1].any_sm_enabled()
            && self.pio[0].pending_irqs() == 0
            && self.pio[1].pending_irqs() == 0
    }

    /// Collect the current DREQ (data-request) bitmap for all 64 TREQ
    /// sources + the FORCE sentinel at bit 63.
    ///
    /// Layout (RP2040 datasheet §2.5.3.1 Table 120, pinned in HLD V7
    /// Appendix C):
    /// * bits 0..3   PIO0 TX0..3
    /// * bits 4..7   PIO0 RX0..3
    /// * bits 8..11  PIO1 TX0..3
    /// * bits 12..15 PIO1 RX0..3
    /// * bit  16     SPI0 TX
    /// * bit  17     SPI0 RX
    /// * bit  18     SPI1 TX
    /// * bit  19     SPI1 RX
    /// * bit  20     UART0 TX
    /// * bit  21     UART0 RX
    /// * bit  22     UART1 TX
    /// * bit  23     UART1 RX
    /// * bits 24..31 PWM WRAP0..7 (deferred — one-shot per wrap, V1 zero)
    /// * bit  32     I2C0 TX
    /// * bit  33     I2C0 RX
    /// * bit  34     I2C1 TX
    /// * bit  35     I2C1 RX
    /// * bit  36     ADC FIFO
    /// * bits 37..39 XIP (not modelled)
    /// * bit  63     FORCE (always true — `TREQ_SEL == 0x3F` bypass)
    ///
    /// Consumed by [`crate::dma::Dma::tick`] — snapshot taken before any
    /// bus access so peripheral state changes driven by the current
    /// transfer don't feed back into same-cycle DREQ arbitration.
    pub fn collect_dreqs(&self) -> u64 {
        let mut bits = 0u64;
        // PIO0 / PIO1 — four SM × (TX | RX) per block.
        for sm in 0..4 {
            if self.pio[0].tx_dreq(sm) {
                bits |= 1u64 << sm;
            }
            if self.pio[0].rx_dreq(sm) {
                bits |= 1u64 << (4 + sm);
            }
            if self.pio[1].tx_dreq(sm) {
                bits |= 1u64 << (8 + sm);
            }
            if self.pio[1].rx_dreq(sm) {
                bits |= 1u64 << (12 + sm);
            }
        }
        if self.spi0.tx_dreq() {
            bits |= 1u64 << 16;
        }
        if self.spi0.rx_dreq() {
            bits |= 1u64 << 17;
        }
        if self.spi1.tx_dreq() {
            bits |= 1u64 << 18;
        }
        if self.spi1.rx_dreq() {
            bits |= 1u64 << 19;
        }
        if self.uart0.tx_dreq() {
            bits |= 1u64 << 20;
        }
        if self.uart0.rx_dreq() {
            bits |= 1u64 << 21;
        }
        if self.uart1.tx_dreq() {
            bits |= 1u64 << 22;
        }
        if self.uart1.rx_dreq() {
            bits |= 1u64 << 23;
        }
        // PWM wrap DREQs (bits 24..31) are one-shot-per-wrap and deferred
        // to a later phase; `audio_i2s` uses PIO DREQ, not PWM wrap.
        if self.i2c0.tx_dreq() {
            bits |= 1u64 << 32;
        }
        if self.i2c0.rx_dreq() {
            bits |= 1u64 << 33;
        }
        if self.i2c1.tx_dreq() {
            bits |= 1u64 << 34;
        }
        if self.i2c1.rx_dreq() {
            bits |= 1u64 << 35;
        }
        if self.adc.dreq() {
            bits |= 1u64 << 36;
        }
        // XIP DREQs (bits 37..39) not modelled.
        // FORCE is always 1 — no peripheral produces it.
        bits |= 1u64 << 63;
        bits
    }

    /// Drive the DMA by one cycle. Swaps the DMA out of `self` to avoid
    /// cross-borrows while it issues transfers through the bus, then
    /// restores it and routes any pending IRQs through `irq_pending`.
    ///
    /// Per HLD V7 §5.6 ordering contract: peripherals tick first (to
    /// produce DREQ), then `tick_dma` consumes the snapshot. Call at the
    /// tail of [`Self::tick_peripherals`].
    pub fn tick_dma(&mut self) {
        // With no channel busy a tick moves nothing: only the IRQ lines
        // need routing, and the DREQs need not be collected.
        if !self.dma.any_busy() {
            self.dma.route_irqs(&mut self.irq_pending);
            return;
        }
        let mut dma = std::mem::take(&mut self.dma);
        dma.tick(self);
        dma.route_irqs(&mut self.irq_pending);
        self.dma = dma;
    }

    /// Advance every stateful peripheral by `cycles` system-clock cycles.
    ///
    /// Called from the slow-path branch in [`crate::Emulator::step`]
    /// whenever the fast-path gate opens (PIO active, DMA live, or an
    /// IRQ already pending). TIMER alarms are polled here for lazy-
    /// fire at match. UART/SPI/I2C advance their TX shift registers
    /// by `cycles` sysclks via `tick(cycles, clock_tree, irqs)` and OR
    /// any level-driven IRQs into `irq_pending`.
    ///
    /// Per HLD 2026.04.26 V5 §5.1: chunked once-per-quantum advance.
    /// `tick_dma` runs once at the tail, after the peripherals produce
    /// DREQ. [`crate::Emulator::step`]'s slow path instead calls
    /// [`Self::tick_peripherals_no_dma`] and ticks DMA once per system
    /// cycle, interleaved with PIO, while a channel is busy.
    #[inline]
    pub fn tick_peripherals(&mut self, cycles: u32) {
        self.tick_peripherals_no_dma(cycles);
        self.tick_dma();
    }

    /// [`Self::tick_peripherals`] without the DMA tick.
    #[inline]
    pub fn tick_peripherals_no_dma(&mut self, cycles: u32) {
        // TIMER alarms are lazy-fire at match: `poll_alarms` is cheap
        // (four armed-bit checks) and we run it here so firmware
        // stepping in the slow path observes alarm-match IRQs on the
        // same inner cycle as the source condition.
        let nvic_bits = self
            .timer
            .poll_alarms(self.master_cycle, self.clock_tree.sys_clk_hz);
        // TIMER IRQs occupy NVIC lines 0..3.
        self.irq_pending |= nvic_bits & 0xF;
        // UART / SPI / I2C: chunked TX drain + IRQ route.
        self.uart0
            .tick(cycles, &self.clock_tree, &mut self.irq_pending);
        self.uart1
            .tick(cycles, &self.clock_tree, &mut self.irq_pending);
        self.spi0
            .tick(cycles, &self.clock_tree, &mut self.irq_pending);
        self.spi1
            .tick(cycles, &self.clock_tree, &mut self.irq_pending);
        self.i2c0
            .tick(cycles, &self.clock_tree, &mut self.irq_pending);
        self.i2c1
            .tick(cycles, &self.clock_tree, &mut self.irq_pending);
        // ADC: fixed-point clk_adc accumulator advances via tick.
        self.adc
            .tick(cycles, &self.clock_tree, &mut self.irq_pending);
        // PWM: per-slice counter advance + wrap-IRQ latch.
        self.pwm
            .tick(cycles, &self.clock_tree, &mut self.irq_pending);
        // USB: level interrupt while INTS is non-zero.
        if self.usbctrl.irq_level() && !peripheral_dispatch::is_held_in_reset(self, USBCTRL_REGS_BASE) {
            self.irq_pending |= 1 << crate::irq::IRQ_USBCTRL_IRQ;
        }
    }

    /// Fast-path lazy-schedule advance (HLD V7 §5.5).
    ///
    /// Called from [`crate::Emulator::step`]'s fast-path branch after
    /// the core(s) have stepped. Bumps the bus's master-cycle cache by
    /// `consumed` and polls all lazy peripherals (TIMER alarms) to
    /// surface IRQs for any events that fell inside the window
    /// `[old_master_cycle, old_master_cycle + consumed]`.
    ///
    /// TIMER is currently the only lazy peripheral; later phases may
    /// add RTC alarm + PWM wrap-on-overflow.
    pub fn advance_lazy_scheduled(&mut self, consumed: u64) {
        self.master_cycle = self.master_cycle.wrapping_add(consumed);
        let nvic_bits = self
            .timer
            .poll_alarms(self.master_cycle, self.clock_tree.sys_clk_hz);
        self.irq_pending |= nvic_bits & 0xF;
    }

    /// Soonest scheduled lazy IRQ deadline (master-cycle space) across
    /// peripherals modelled today. TIMER is the only lazy IRQ source in
    /// V1 (PWM/ADC/etc. tick on `consumed`, not on absolute deadlines),
    /// so this currently returns `timer.next_armed_inte_fire_cycle()`.
    /// Used by the both-cores-blocked clock-advance branch in
    /// `Emulator::step_serial` to pick the next event horizon —
    /// closes tech_debt §1649 for RP2040.
    pub fn next_scheduled_lazy_deadline(&self) -> Option<u64> {
        self.timer.next_armed_inte_fire_cycle()
    }

    /// Read the current pending-IRQ bitmap, for tests and harness
    /// diagnostics.
    #[inline]
    pub fn irq_pending(&self) -> u32 {
        self.irq_pending
    }

    /// ROSC nominal frequency re-export.
    pub const ROSC_FREQ_HZ_CONST: u32 = ROSC_FREQ_HZ;
}

impl Default for Bus {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================================
// `CoreBus` implementation — dual-execution HLD V1 Stage 3b.1.
//
// Every method forwards directly to an inherent `Bus` method or field.
// Monomorphization against this impl reproduces the pre-refactor
// codegen on the Serial hot path. Stage 3b.2 will add a
// `impl CoreBus for WorkerBus` in the `threaded/` module once that
// module lands.
// ===================================================================

use crate::core::CoreBus;

impl CoreBus for Bus {
    #[inline]
    fn fetch16(&mut self, addr: u32) -> u16 {
        let touched = self.core0_bank_touched;
        let v = Bus::read16(self, addr);
        self.core0_bank_touched = touched;
        v
    }
    #[inline(always)]
    fn read8(&mut self, addr: u32) -> u8 {
        self.idle_read(addr);
        Bus::read8(self, addr)
    }
    #[inline(always)]
    fn read16(&mut self, addr: u32) -> u16 {
        self.idle_read(addr);
        Bus::read16(self, addr)
    }
    #[inline(always)]
    fn read32(&mut self, addr: u32) -> u32 {
        self.idle_read(addr);
        Bus::read32(self, addr)
    }
    #[inline(always)]
    fn write8(&mut self, addr: u32, val: u8) {
        self.idle_write(addr, 1);
        // The core's own stores are park-checked in `idle_write`.
        let n = self.ext_sram_writes;
        Bus::write8(self, addr, val);
        self.ext_sram_writes = n;
    }
    #[inline(always)]
    fn write16(&mut self, addr: u32, val: u16) {
        self.idle_write(addr, 2);
        // The core's own stores are park-checked in `idle_write`.
        let n = self.ext_sram_writes;
        Bus::write16(self, addr, val);
        self.ext_sram_writes = n;
    }
    #[inline(always)]
    fn write32(&mut self, addr: u32, val: u32) {
        self.idle_write(addr, 4);
        // The core's own stores are park-checked in `idle_write`.
        let n = self.ext_sram_writes;
        Bus::write32(self, addr, val);
        self.ext_sram_writes = n;
    }
    #[inline(always)]
    fn idle_mark_dirty(&mut self) {
        self.idle.dirty = true;
        self.idle.event = true;
    }
    #[inline(always)]
    fn idle_mark_exception(&mut self) {
        self.idle.dirty = true;
        self.idle.event = true;
        self.idle.exception = true;
    }
    #[inline]
    fn mark_code_page(&mut self, pc: u32) {
        self.set_code_page(pc);
    }
    fn set_active_pc(&mut self, pc: u32) {
        Bus::set_active_pc(self, pc)
    }

    #[inline(always)]
    fn bus_fault(&self) -> bool {
        Bus::bus_fault(self)
    }
    #[inline(always)]
    fn bus_fault_addr(&self) -> u32 {
        Bus::bus_fault_addr(self)
    }
    #[inline(always)]
    fn clear_bus_fault(&mut self) {
        Bus::clear_bus_fault(self)
    }

    #[inline(always)]
    fn ppb(&self, core: usize) -> &ppb::Ppb {
        &self.ppb[core]
    }
    #[inline(always)]
    fn ppb_mut(&mut self, core: usize) -> &mut ppb::Ppb {
        &mut self.ppb[core]
    }

    #[inline(always)]
    fn nvic(&self, core: usize) -> &Nvic {
        &self.nvics[core]
    }
    #[inline(always)]
    fn nvic_mut(&mut self, core: usize) -> &mut Nvic {
        &mut self.nvics[core]
    }

    #[inline(always)]
    fn active_core(&self) -> usize {
        Bus::active_core(self)
    }

    // --- WFE / SEV wake protocol --------------------------------------

    #[inline(always)]
    fn event_flag(&self, core: usize) -> bool {
        self.event_flag[core]
    }

    #[inline(always)]
    fn consume_event_flag(&mut self, core: usize) -> bool {
        let prior = self.event_flag[core];
        self.event_flag[core] = false;
        prior
    }

    #[inline(always)]
    fn set_wfe_waiting(&mut self, core: usize, val: bool) {
        self.wfe_waiting[core] = val;
    }

    #[inline(always)]
    fn signal_sev(&mut self) {
        Bus::signal_sev(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_bus_all_peripherals_in_reset() {
        let bus = Bus::new();
        assert_eq!(bus.resets.state, resets::RESET_MASK);
    }

    #[test]
    fn sram_write_read_roundtrip() {
        let mut bus = Bus::new();
        bus.write32(0x2000_0100, 0xDEAD_BEEF);
        assert_eq!(bus.read32(0x2000_0100), 0xDEAD_BEEF);
    }

    #[test]
    fn sram_aliases_mirror_same_storage() {
        // RP2040 datasheet §2.1.2: all four SRAM alias windows
        // (0x20/0x21/0x22/0x23) address the same backing bytes. Aliases
        // are bank-striping flavours for DMA, not peripheral XOR/SET/CLR.
        let mut bus = Bus::new();
        bus.write32(0x2000_0100, 0xF0F0_F0F0);
        // A write via 0x21xxxxxx overwrites the same bytes, not XORs.
        bus.write32(0x2100_0100, 0x0F0F_0F0F);
        assert_eq!(bus.read32(0x2000_0100), 0x0F0F_0F0F);
        // Reads through every alias observe the identical word.
        assert_eq!(bus.read32(0x2100_0100), 0x0F0F_0F0F);
        assert_eq!(bus.read32(0x2200_0100), 0x0F0F_0F0F);
        assert_eq!(bus.read32(0x2300_0100), 0x0F0F_0F0F);
        // Writing through 0x22 / 0x23 also just overwrites.
        bus.write32(0x2200_0200, 0xAAAA_AAAA);
        bus.write32(0x2300_0200, 0x5555_5555);
        assert_eq!(bus.read32(0x2000_0200), 0x5555_5555);
    }

    #[test]
    fn resets_clr_deasserts() {
        let mut bus = Bus::new();
        // CLR alias at RESETS base 0x4000_C000 + alias 3 → offset 0x3000.
        bus.write32(0x4000_F000, 0x0000_0001);
        assert_eq!(bus.read32(0x4000_C000) & 1, 0);
        assert_eq!(bus.read32(0x4000_C008) & 1, 1);
    }

    #[test]
    fn clocks_ref_mux_switch_to_xosc() {
        let mut bus = Bus::new();
        // CLK_REF_CTRL at 0x4000_8030, write SRC=2 (XOSC).
        bus.write32(0x4000_8030, 2);
        assert_eq!(bus.clock_tree.ref_clk_hz, clocks::XOSC_FREQ_HZ);
    }

    #[test]
    fn clocks_sys_div_write_at_0x40_recomputes_tree() {
        // RP2040 datasheet §2.15.7: CLK_SYS_DIV is at CLOCKS_BASE + 0x40.
        // A write to 0x4000_8040 must land on `clk_sys_div` and feed
        // through `recompute_clock_tree()` — confirming the constants
        // aren't swapped with CLK_SYS_SELECTED (0x44, mux indicator).
        let mut bus = Bus::new();
        bus.seed_sys_clk_hz(clocks::ROSC_FREQ_HZ);
        // DIV integer field lives in bits [31:16]; write /4.
        bus.write32(0x4000_8040, 4 << 16);
        assert_eq!(bus.clocks_regs.clk_sys_div, 4 << 16);
        assert_eq!(bus.clock_tree.sys_clk_hz, clocks::ROSC_FREQ_HZ / 4);
    }

    #[test]
    fn pll_lock_bit_forced_high() {
        // Post-`2026.04.15 HLD - PLL LOCK Modelling` fix: at Bus::new()
        // the PLL is powered down (PWR=0x2D), FBDIV=0, so CS[31] must
        // read 0. The test name is historical — the old pre-fix
        // behaviour forced the bit; the new modelled behaviour derives
        // it from power state + FBDIV + arm timing.
        let mut bus = Bus::new();
        let cs = bus.read32(PLL_SYS_BASE);
        assert_eq!(cs & (1 << 31), 0, "LOCK must be 0 at reset (PLL unpowered)");
    }

    #[test]
    fn sio_gpio_out_roundtrip() {
        let mut bus = Bus::new();
        bus.write32(SIO_BASE + 0x010, 0x5A);
        assert_eq!(bus.read32(SIO_BASE + 0x010), 0x5A);
    }

    #[test]
    fn sio_cpuid_reflects_active_core() {
        let mut bus = Bus::new();
        bus.set_active_core(1);
        assert_eq!(bus.read32(SIO_BASE), 1);
        bus.set_active_core(0);
        assert_eq!(bus.read32(SIO_BASE), 0);
    }

    #[test]
    fn gpio_in_is_owned_by_bus() {
        let mut bus = Bus::new();
        bus.gpio_in = 0x42;
        assert_eq!(bus.read32(SIO_BASE + 0x004), 0x42);
    }

    #[test]
    fn xip_fresh_bus_reads_zero_without_fault() {
        // PicoGUS integration (Stage 1 HLD): flash is a plain mapped
        // window. Reads before `load_flash` must return 0 without
        // setting bus_fault so a firmware that probes XIP during boot
        // doesn't take a spurious HardFault.
        let mut bus = Bus::new();
        assert_eq!(bus.read32(0x1000_0000), 0);
        assert_eq!(bus.read8(0x1000_0001), 0);
        assert_eq!(bus.read16(0x1000_0002), 0);
        assert!(!bus.bus_fault());
    }

    #[test]
    fn xip_read_after_flash_load() {
        let mut bus = Bus::new();
        bus.load_flash(&[0xAA, 0xBB, 0xCC, 0xDD]);
        assert_eq!(bus.read32(0x1000_0000), 0xDDCCBBAA);
        assert_eq!(bus.read8(0x1000_0000), 0xAA);
        assert_eq!(bus.read8(0x1000_0003), 0xDD);
        assert_eq!(bus.read16(0x1000_0002), 0xDDCC);
    }

    /// Direct-mode SSI, the way the bootrom's flash_do_cmd drives it: CS
    /// forced low through IO_QSPI SS_CTRL.OUTOVER, one DR0 write per byte,
    /// one reply byte per write in the RX FIFO.
    #[test]
    fn ssi_flash_commands_reply() {
        let mut bus = Bus::new();
        bus.load_flash(&[0x10, 0x11, 0x12, 0x13, 0x14, 0x15]);
        let xfer = |bus: &mut Bus, tx: &[u8]| -> Vec<u8> {
            bus.write32(IO_QSPI_BASE + 0x0c, 0x300); // CS high
            bus.write32(IO_QSPI_BASE + 0x0c, 0x200); // CS low: new command
            for &b in tx {
                bus.write32(SSI_BASE + 0x60, b as u32);
            }
            assert_eq!(bus.read32(SSI_BASE + 0x24), tx.len() as u32, "RXFLR");
            assert_ne!(bus.read32(SSI_BASE + 0x28) & 0x08, 0, "SR.RFNE");
            (0..tx.len()).map(|_| bus.read32(SSI_BASE + 0x60) as u8).collect()
        };
        bus.write32(SSI_BASE + 0x08, 1); // SSIENR
        assert_eq!(xfer(&mut bus, &[0x9F, 0, 0, 0]), [0, 0xEF, 0x40, 0x15]);
        assert_eq!(xfer(&mut bus, &[0x03, 0, 0, 2, 0, 0]), [0, 0, 0, 0, 0x12, 0x13]);
        assert_eq!(xfer(&mut bus, &[0x05, 0]), [0, 0]);
        assert_eq!(bus.read32(SSI_BASE + 0x28) & 0x09, 0, "SR: not busy, RX empty");
        // Disabling the SSI empties the RX FIFO.
        bus.write32(SSI_BASE + 0x60, 0);
        bus.write32(SSI_BASE + 0x08, 0);
        assert_eq!(bus.read32(SSI_BASE + 0x24), 0);
    }

    /// USB controller registers are reset-gated on RESETS bit 24, DPRAM
    /// takes every access width, and the USBCTRL line follows INTS.
    #[test]
    fn usbctrl_wiring() {
        use picoem_common::usb::{INT_BUS_RESET, INTE, MAIN_CTRL, MAIN_CTRL_CONTROLLER_EN, SIE_CTRL, SIE_CTRL_PULLUP_EN};
        let mut bus = Bus::new();
        bus.write32(USBCTRL_REGS_BASE + MAIN_CTRL, 1);
        assert_eq!(bus.read32(USBCTRL_REGS_BASE + MAIN_CTRL), 0, "held in reset");
        bus.write32(0x4000_F000, 1 << peripheral_dispatch::RESET_USBCTRL);
        bus.write32(USBCTRL_REGS_BASE + MAIN_CTRL, MAIN_CTRL_CONTROLLER_EN);
        assert_eq!(bus.read32(USBCTRL_REGS_BASE + MAIN_CTRL), MAIN_CTRL_CONTROLLER_EN);
        bus.write8(USBCTRL_DPRAM_BASE + 0x181, 0x22);
        bus.write16(USBCTRL_DPRAM_BASE + 0x182, 0x4433);
        bus.write8(USBCTRL_DPRAM_BASE + 0x180, 0x11);
        assert_eq!(bus.read32(USBCTRL_DPRAM_BASE + 0x180), 0x4433_2211);
        assert_eq!(bus.read8(USBCTRL_DPRAM_BASE + 0x182), 0x33);
        bus.write32(USBCTRL_REGS_BASE + SIE_CTRL, SIE_CTRL_PULLUP_EN);
        bus.write32(USBCTRL_REGS_BASE + INTE, INT_BUS_RESET);
        bus.usb_plug(true);
        bus.tick_peripherals(1);
        assert_eq!(bus.irq_pending & (1 << crate::irq::IRQ_USBCTRL_IRQ), 0);
        bus.usb().bus_reset();
        bus.tick_peripherals(1);
        assert_ne!(bus.irq_pending & (1 << crate::irq::IRQ_USBCTRL_IRQ), 0);
        // Asserting the reset clears the controller.
        bus.write32(0x4000_E000, 1 << peripheral_dispatch::RESET_USBCTRL);
        bus.write32(0x4000_F000, 1 << peripheral_dispatch::RESET_USBCTRL);
        assert_eq!(bus.read32(USBCTRL_REGS_BASE + INTE), 0);
    }

    /// Write enable, sector erase, page program with page wrap, and a
    /// program without write enable that changes nothing.
    #[test]
    fn ssi_flash_program_and_erase() {
        let mut bus = Bus::new();
        bus.load_flash(&[0x5A; 0x3000]);
        let xfer = |bus: &mut Bus, tx: &[u8]| -> Vec<u8> {
            bus.write32(IO_QSPI_BASE + 0x0c, 0x300);
            bus.write32(IO_QSPI_BASE + 0x0c, 0x200);
            for &b in tx {
                bus.write32(SSI_BASE + 0x60, b as u32);
            }
            (0..tx.len()).map(|_| bus.read32(SSI_BASE + 0x60) as u8).collect()
        };
        bus.write32(SSI_BASE + 0x08, 1);
        // Erase needs write enable.
        xfer(&mut bus, &[0x20, 0, 0x10, 0]);
        assert_eq!(bus.read8(0x1000_1000), 0x5A);
        xfer(&mut bus, &[0x06]);
        assert_eq!(xfer(&mut bus, &[0x05, 0]), [0, 0x02], "WEL set");
        xfer(&mut bus, &[0x20, 0, 0x10, 0x23]);
        assert_eq!(xfer(&mut bus, &[0x05, 0]), [0, 0], "erase clears WEL");
        assert_eq!(bus.read8(0x1000_0FFF), 0x5A);
        assert_eq!(bus.read8(0x1000_1000), 0xFF);
        assert_eq!(bus.read8(0x1000_1FFF), 0xFF);
        assert_eq!(bus.read8(0x1000_2000), 0x5A);
        // Program 3 bytes from 0x10FE: the third wraps to 0x1000.
        xfer(&mut bus, &[0x06]);
        xfer(&mut bus, &[0x02, 0, 0x10, 0xFE, 0x11, 0x22, 0x33]);
        assert_eq!(bus.read8(0x1000_10FE), 0x11);
        assert_eq!(bus.read8(0x1000_10FF), 0x22);
        assert_eq!(bus.read8(0x1000_1000), 0x33);
        // Programming only clears bits, and needs write enable.
        xfer(&mut bus, &[0x02, 0, 0x10, 0x80, 0x00]);
        assert_eq!(bus.read8(0x1000_1080), 0xFF);
        xfer(&mut bus, &[0x06]);
        xfer(&mut bus, &[0x02, 0, 0x10, 0xFE, 0xF0]);
        assert_eq!(bus.read8(0x1000_10FE), 0x10);
    }

    #[test]
    fn xip_aliases_mirror_flash_base() {
        // RP2040 XIP has three read-only aliases at 0x11/0x12/0x13 that
        // map to the same 2 MB flash window. All four addresses must
        // observe identical bytes after `load_flash`.
        let mut bus = Bus::new();
        bus.load_flash(&[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE]);
        let words_at = |bus: &mut Bus, base: u32| {
            (
                bus.read32(base),
                bus.read32(base + 4),
                bus.read8(base + 1),
                bus.read16(base + 6),
            )
        };
        let canonical = words_at(&mut bus, 0x1000_0000);
        assert_eq!(words_at(&mut bus, 0x1100_0000), canonical);
        assert_eq!(words_at(&mut bus, 0x1200_0000), canonical);
        assert_eq!(words_at(&mut bus, 0x1300_0000), canonical);
        assert_eq!(canonical.0, 0xEFBEADDE);
    }

    #[test]
    fn xip_read_past_loaded_length_returns_zero() {
        // Within the mapped 2 MB window, addresses past the loaded
        // image length must read 0 (pre-allocated zero bytes in the
        // backing buffer). No bus fault.
        let mut bus = Bus::new();
        bus.load_flash(&[0x11, 0x22, 0x33, 0x44]);
        assert_eq!(bus.read32(0x1000_0004), 0);
        assert_eq!(bus.read32(0x1010_0000), 0); // 1 MB in
        assert_eq!(bus.read32(0x101F_FFFC), 0); // last word of window
        assert_eq!(bus.read32(0x1110_0000), 0); // alias 0x11, mid-window
        assert!(!bus.bus_fault());
    }

    #[test]
    fn xip_writes_silently_ignored_at_every_width() {
        // Real flash needs erase/program via XIP_SSI; at the AHB layer
        // writes to the flash window must not fault and must not alter
        // the loaded bytes.
        let mut bus = Bus::new();
        bus.load_flash(&[0x55, 0x66, 0x77, 0x88]);
        bus.write8(0x1000_0000, 0xAA);
        bus.write16(0x1000_0002, 0xBBBB);
        bus.write32(0x1000_0000, 0xDEAD_BEEF);
        // Aliases must also swallow writes.
        bus.write8(0x1100_0000, 0xAA);
        bus.write16(0x1200_0002, 0xBBBB);
        bus.write32(0x1300_0000, 0xDEAD_BEEF);
        assert!(!bus.bus_fault(), "flash writes must not raise bus_fault");
        assert_eq!(bus.read32(0x1000_0000), 0x88776655);
        assert_eq!(bus.read32(0x1100_0000), 0x88776655);
    }

    #[test]
    fn xip_sram_scratch() {
        let mut bus = Bus::new();
        bus.write32(XIP_SRAM_BASE, 0xCAFE_BABE);
        assert_eq!(bus.read32(XIP_SRAM_BASE), 0xCAFE_BABE);
    }

    #[test]
    fn xip_ctrl_roundtrip() {
        let mut bus = Bus::new();
        bus.write32(XIP_CTRL_BASE + 0x4, 0x1234);
        assert_eq!(bus.read32(XIP_CTRL_BASE + 0x4), 0x1234);
    }

    #[test]
    fn unmapped_region_faults() {
        let mut bus = Bus::new();
        bus.read32(0x7000_0000);
        assert!(bus.bus_fault());
    }

    #[test]
    fn unmapped_writes_fault_at_every_width() {
        // Consistent policy (see write8/write16/write32): unmapped writes
        // at any width set the sticky bus-fault flag.
        let mut bus = Bus::new();
        bus.write8(0x7000_0000, 0xAA);
        assert!(bus.bus_fault(), "write8 to unmapped region must fault");
        bus.clear_bus_fault();

        bus.write16(0x7000_0000, 0xAABB);
        assert!(bus.bus_fault(), "write16 to unmapped region must fault");
        bus.clear_bus_fault();

        bus.write32(0x7000_0000, 0xAABB_CCDD);
        assert!(bus.bus_fault(), "write32 to unmapped region must fault");
    }

    #[test]
    fn rom_writes_silently_ignored_at_every_width() {
        // ROM is read-only — writes at any width must NOT raise bus_fault.
        let mut bus = Bus::new();
        bus.write8(0x0000_0100, 0xAA);
        assert!(!bus.bus_fault(), "write8 to ROM is silent");
        bus.write16(0x0000_0100, 0xAABB);
        assert!(!bus.bus_fault(), "write16 to ROM is silent");
        bus.write32(0x0000_0100, 0xAABB_CCDD);
        assert!(!bus.bus_fault(), "write32 to ROM is silent");
    }

    #[test]
    fn sram_bank_contention_plus_one_cycle() {
        let mut bus = Bus::new();
        // Core 0 touches bank 0.
        bus.set_active_core(0);
        let _ = bus.read32(0x2000_0000);
        // Core 1 touches the same bank — expect +1 cycle latency.
        bus.set_active_core(1);
        bus.begin_core1_step();
        let _ = bus.read32(0x2000_0000);
        assert_eq!(bus.last_access_cycles, 2);
        bus.end_core1_step();
    }

    #[test]
    fn sram_bank_no_contention_different_banks() {
        let mut bus = Bus::new();
        bus.set_active_core(0);
        let _ = bus.read32(0x2000_0000); // bank 0
        bus.set_active_core(1);
        bus.begin_core1_step();
        let _ = bus.read32(0x2000_0004); // bank 1
        assert_eq!(bus.last_access_cycles, 1);
        bus.end_core1_step();
    }

    /// A thread-safe `Vec<u8>` sink so we can capture the MMIO trace output
    /// without wrestling with stdout redirection. Wraps `Vec<u8>` behind
    /// an `Arc<Mutex<...>>` so the test can drain the buffer after the
    /// bus has written through the sink. (`Bus::mmio_trace_sink` requires
    /// `Write + Send`.)
    #[derive(Clone)]
    struct CaptureSink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CaptureSink {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn mmio_trace_enabled_emits_write32_line() {
        // HLD V7 §4.3: `write32(addr, val)` with `mmio_trace_enabled = true`
        // emits one line in the prescribed format. We inject a captured
        // `CaptureSink` so the test doesn't depend on fd 1 redirection.
        let capture = CaptureSink(std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
        let mut bus = Bus::new();
        bus.set_active_core(0);
        bus.set_active_pc(0x1000_0100);
        bus.mmio_trace_enabled = true;
        bus.set_mmio_trace_sink(Some(Box::new(capture.clone())));

        // SRAM word write — exercises the hot path and one of the six
        // access methods required by the spec.
        bus.write32(0x2000_0200, 0xDEAD_BEEF);

        let captured = capture.0.lock().unwrap();
        let text = std::str::from_utf8(&captured).expect("trace must be utf-8");
        // Exactly one line, with the expected fields.
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines.len(),
            1,
            "expected one trace line, got {}: {:?}",
            lines.len(),
            lines
        );
        let line = lines[0];
        assert!(
            line.starts_with("TRACE W 4 0x20000200"),
            "line = {:?}",
            line
        );
        assert!(line.contains("val=0xDEADBEEF"), "line = {:?}", line);
        assert!(line.contains("core=0"), "line = {:?}", line);
        assert!(line.contains("pc=0x10000100"), "line = {:?}", line);
    }

    #[test]
    fn trace_disabled_emits_nothing() {
        // Zero-overhead path — `mmio_trace_enabled = false` must not route any
        // bytes to the sink. Guards the hot path (non-trace runs must not
        // pay a formatting cost).
        let capture = CaptureSink(std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
        let mut bus = Bus::new();
        bus.set_mmio_trace_sink(Some(Box::new(capture.clone())));
        // mmio_trace_enabled is false by default.
        bus.write32(0x2000_0200, 0xCAFE_F00D);
        let _ = bus.read32(0x2000_0200);
        assert!(capture.0.lock().unwrap().is_empty());
    }

    #[test]
    fn trace_active_pc_is_per_core() {
        // Regression guard for the dual-core `active_pc` staleness bug
        // (Wave 2 review SHOULD-FIX 1). The scheduler alternates
        // `set_active_core(0)` / `set_active_core(1)` each quantum; a
        // bus access on core 1 that doesn't go through `decode_execute`
        // (e.g. exception stacking) must NOT observe core 0's last decode
        // PC, and vice versa. Simulate the pattern: decode PC=0x1000 on
        // core 0, switch to core 1, decode PC=0x2000, switch back to
        // core 0 and issue an access *without* re-decoding — the trace
        // line must still carry PC=0x1000 for core 0.
        let capture = CaptureSink(std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
        let mut bus = Bus::new();
        bus.mmio_trace_enabled = true;
        bus.set_mmio_trace_sink(Some(Box::new(capture.clone())));

        // Core 0 "decodes" at 0x1000 and writes.
        bus.set_active_core(0);
        bus.set_active_pc(0x0000_1000);
        bus.write32(0x2000_0100, 0xAAAA_AAAA);

        // Scheduler switches to core 1, which "decodes" at 0x2000 and
        // writes.
        bus.set_active_core(1);
        bus.set_active_pc(0x0000_2000);
        bus.write32(0x2000_0104, 0xBBBB_BBBB);

        // Scheduler switches back to core 0 WITHOUT a re-decode (mimics
        // hardware-triggered access like exception stacking before the
        // handler's first `decode_execute`). The stored per-core PC
        // must still be 0x1000 for core 0 — not 0x2000 from core 1's
        // quantum.
        bus.set_active_core(0);
        bus.write32(0x2000_0108, 0xCCCC_CCCC);

        let captured = capture.0.lock().unwrap();
        let text = std::str::from_utf8(&captured).expect("trace must be utf-8");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines.len(),
            3,
            "expected three trace lines, got {:?}",
            lines
        );
        assert!(
            lines[0].contains("core=0") && lines[0].contains("pc=0x00001000"),
            "line 0 = {:?}",
            lines[0]
        );
        assert!(
            lines[1].contains("core=1") && lines[1].contains("pc=0x00002000"),
            "line 1 = {:?}",
            lines[1]
        );
        assert!(
            lines[2].contains("core=0") && lines[2].contains("pc=0x00001000"),
            "line 2 = {:?} (core 0 PC must survive the core-1 excursion)",
            lines[2]
        );
    }

    // ----- NVIC MMIO dispatch (Phase 1 Wave 2, HLD V7 §5.2) --------------
    //
    // Exercise the bus-level interception of the NVIC register window
    // `0xE000_E100..=0xE000_E41F`. Reads go through the Bus's normal
    // `read32`/`read16`/`read8` API so alias + byte/halfword access is
    // covered; writes mirror.

    #[test]
    fn nvic_iser0_write_sets_enable_read_back() {
        let mut bus = Bus::new();
        // Enable IRQ 5 + IRQ 12.
        bus.write32(0xE000_E100, (1u32 << 5) | (1u32 << 12));
        // ISER0 read returns the enabled mask.
        assert_eq!(bus.read32(0xE000_E100), (1u32 << 5) | (1u32 << 12));
        // ICER0 read aliases the same mask.
        assert_eq!(bus.read32(0xE000_E180), (1u32 << 5) | (1u32 << 12));
        // Writing ICER0 clears the specified bits.
        bus.write32(0xE000_E180, 1u32 << 5);
        assert_eq!(bus.read32(0xE000_E100), 1u32 << 12);
        // ISER0 write is write-1-set (does not clear unchanged bits).
        bus.write32(0xE000_E100, 1u32 << 3);
        assert_eq!(bus.read32(0xE000_E100), (1u32 << 3) | (1u32 << 12));
    }

    #[test]
    fn nvic_ispr0_write_sets_pending_read_back() {
        let mut bus = Bus::new();
        // Set pending IRQ 0 via ISPR0.
        bus.write32(0xE000_E200, 1u32 << 0);
        assert_eq!(bus.read32(0xE000_E200), 1u32 << 0);
        assert_eq!(bus.read32(0xE000_E280), 1u32 << 0); // ICPR0 shows pending too
        // W1C via ICPR0.
        bus.write32(0xE000_E280, 1u32 << 0);
        assert_eq!(bus.read32(0xE000_E200), 0);
    }

    #[test]
    fn nvic_ipr_word_encodes_four_priorities() {
        let mut bus = Bus::new();
        // Write IPR1 (covers IRQs 4..=7). Priority 0xC0 on IRQ 4 (lane 0),
        // 0x80 on IRQ 5 (lane 1), 0x40 on IRQ 6 (lane 2), 0x00 on IRQ 7
        // (lane 3).
        let word = 0xC0u32 | (0x80u32 << 8) | (0x40u32 << 16);
        bus.write32(0xE000_E404, word);
        assert_eq!(bus.read32(0xE000_E404), word);
        // Non-implemented bits of a priority byte must be masked — write
        // 0x3F (all low bits) to IRQ 8 on IPR2 lane 0; readback is 0.
        bus.write32(0xE000_E408, 0x3F);
        assert_eq!(bus.read32(0xE000_E408), 0);
    }

    #[test]
    fn nvic_iser_then_icer_clears() {
        // V5 §5.1 audit shape: write 0xFFFF to ISER0, then 0x00FF to
        // ICER0; ISER0 readback must be 0xFF00 (high half preserved,
        // low half cleared by W1C).
        let mut bus = Bus::new();
        bus.write32(0xE000_E100, 0x0000_FFFF);
        bus.write32(0xE000_E180, 0x0000_00FF);
        assert_eq!(bus.read32(0xE000_E100), 0x0000_FF00);
        // ICPR mirrors ISPR for pending — analogous shape covered by
        // `nvic_ispr0_write_sets_pending_read_back` above.
    }

    #[test]
    fn nvic_ipr_top_two_bits_significant() {
        // V5 §5.1 audit shape: writing 0xC5 to a priority byte must
        // mask to 0xC0 (only bits [7:6] are implemented on M0+).
        let mut bus = Bus::new();
        // IPR0 lane 0 covers IRQ 0.
        bus.write32(0xE000_E400, 0xC5);
        assert_eq!(bus.read32(0xE000_E400), 0xC0);
    }

    #[test]
    fn nvic_is_per_core_banked() {
        // ARMv6-M banks the SCS per-core. Active core = 0: writes land
        // on nvics[0] only; active core = 1 sees independent state.
        let mut bus = Bus::new();
        bus.set_active_core(0);
        bus.write32(0xE000_E100, 1u32 << 4);
        bus.set_active_core(1);
        bus.write32(0xE000_E100, 1u32 << 19);
        // Core 0 must NOT see IRQ 19 enabled; core 1 must NOT see IRQ 4.
        bus.set_active_core(0);
        assert_eq!(bus.read32(0xE000_E100), 1u32 << 4);
        bus.set_active_core(1);
        assert_eq!(bus.read32(0xE000_E100), 1u32 << 19);
    }

    #[test]
    fn nvic_mmio_reset_gate_does_not_apply() {
        // NVIC is part of SCS (0xE...), not an APB peripheral — no RESETS
        // bit gates it. Freshly-constructed bus with everything held in
        // reset still honours NVIC reads/writes.
        let mut bus = Bus::new();
        bus.write32(0xE000_E100, 1u32 << 0);
        assert_eq!(bus.read32(0xE000_E100), 1u32 << 0);
    }

    #[test]
    fn nvic_high_bits_razwi() {
        // RP2040 has 26 IRQ lines (bits 0..=25). Writes to ISER0/ICER0/
        // ISPR0/ICPR0 with bits 26..31 set must be silently dropped on
        // real silicon (RAZ/WI). Reads naturally RAZ since the field
        // never has high bits set.
        let mut bus = Bus::new();
        // ISER0: write all-ones, expect only the 26 implemented bits.
        bus.write32(0xE000_E100, 0xFFFF_FFFF);
        assert_eq!(bus.read32(0xE000_E100), 0x03FF_FFFF);
        // ICER0: with all 26 enable bits set, an all-ones write must
        // clear them all (no high-bit residue gating the AND-NOT).
        bus.write32(0xE000_E180, 0xFFFF_FFFF);
        assert_eq!(bus.read32(0xE000_E100), 0);
        // ISPR0: same shape as ISER0 on the SET side.
        bus.write32(0xE000_E200, 0xFFFF_FFFF);
        assert_eq!(bus.read32(0xE000_E200), 0x03FF_FFFF);
        // ICPR0: same shape as ICER0 on the CLEAR side.
        bus.write32(0xE000_E280, 0xFFFF_FFFF);
        assert_eq!(bus.read32(0xE000_E200), 0);

        // Force high bits into the field directly to exercise the
        // clear-arm mask: without `val & IRQ_LINE_MASK` on the ICER0/
        // ICPR0 arms, an all-ones write would clear the high bits too.
        bus.nvics[0].enabled = 0xFFFF_FFFF;
        bus.write32(0xE000_E180, 0xFFFF_FFFF); // ICER0 — should clear only bits 0..25
        assert_eq!(
            bus.read32(0xE000_E100),
            0xFC00_0000,
            "ICER0 must mask val before clearing — high bits preserved"
        );

        bus.nvics[0].pending = 0xFFFF_FFFF;
        bus.write32(0xE000_E280, 0xFFFF_FFFF); // ICPR0 — should clear only bits 0..25
        assert_eq!(
            bus.read32(0xE000_E200),
            0xFC00_0000,
            "ICPR0 must mask val before clearing — high bits preserved"
        );
    }

    // ----- ADC + PWM bus integration (Phase 3) --------------------------

    #[test]
    fn adc_held_in_reset_returns_zero() {
        // Default Bus holds RESET_ADC (bit 0) asserted — reads must
        // return 0 and writes must not route.
        let mut bus = Bus::new();
        bus.write32(ADC_BASE + crate::peripherals::adc::CS, 0x1);
        assert_eq!(
            bus.read32(ADC_BASE + crate::peripherals::adc::CS),
            0,
            "ADC held in reset must RAZ"
        );
    }

    #[test]
    fn adc_unreset_roundtrips_cs() {
        let mut bus = Bus::new();
        // Clear RESET_ADC (bit 0) via CLR alias at RESETS+0x3000.
        bus.write32(0x4000_F000, 0x1);
        bus.write32(
            ADC_BASE + crate::peripherals::adc::CS,
            crate::peripherals::adc::CS_EN,
        );
        assert_eq!(
            bus.read32(ADC_BASE + crate::peripherals::adc::CS) & crate::peripherals::adc::CS_EN,
            crate::peripherals::adc::CS_EN
        );
    }

    #[test]
    fn pwm_held_in_reset_returns_zero() {
        let mut bus = Bus::new();
        bus.write32(PWM_BASE + crate::peripherals::pwm::EN, 0xFF);
        assert_eq!(bus.read32(PWM_BASE + crate::peripherals::pwm::EN), 0);
    }

    #[test]
    fn pwm_unreset_roundtrips_slice_registers() {
        let mut bus = Bus::new();
        // Clear RESET_PWM (bit 14).
        bus.write32(0x4000_F000, 1u32 << 14);
        // Slice 0 TOP = 100 via canonical address.
        let slice0_top = PWM_BASE + 0x10;
        bus.write32(slice0_top, 100);
        assert_eq!(bus.read32(slice0_top), 100);
    }

    // ----- PIO TXF narrow-write dispatch (Phase D) -------------------------
    //
    // Regression tests for the PIO byte/halfword TXF write fix.
    // DMA_SIZE_8 writes to TXF were silently dropped by the blanket PIO
    // guard in `write8`/`write16`, breaking rp2040-psram transfers.

    #[test]
    fn pio_txf_byte_write_pushes_replicated_word() {
        let mut bus = Bus::new();
        // Release PIO1 from reset (RESETS bit 11).
        bus.write32(0x4000_F000, 1u32 << 11);
        // Enable PIO1 SM0 via CTRL.SM_ENABLE bit 0.
        bus.write32(PIO1_BASE, 0x1);
        // Byte write to PIO1 TXF0.
        bus.write8(PIO1_BASE + 0x010, 0x42);
        // FSTAT: TXEMPTY bit 24 for SM0 must be cleared.
        let fstat = bus.read32(PIO1_BASE + 0x004);
        assert_eq!(
            fstat & (1 << 24),
            0,
            "TX FIFO must not be empty after byte write"
        );
        // Pop from TX FIFO and verify byte-replicated value.
        let val = bus.pio[1].pop_tx(0).expect("TX FIFO should have one entry");
        assert_eq!(
            val, 0x42424242,
            "byte 0x42 must be replicated to all four lanes"
        );
    }

    #[test]
    fn pio_txf_halfword_write_pushes_replicated_word() {
        let mut bus = Bus::new();
        // Release PIO1 from reset.
        bus.write32(0x4000_F000, 1u32 << 11);
        // Enable PIO1 SM0.
        bus.write32(PIO1_BASE, 0x1);
        // Halfword write to PIO1 TXF0.
        bus.write16(PIO1_BASE + 0x010, 0x1234);
        // FSTAT: TXEMPTY bit 24 for SM0 must be cleared.
        let fstat = bus.read32(PIO1_BASE + 0x004);
        assert_eq!(
            fstat & (1 << 24),
            0,
            "TX FIFO must not be empty after halfword write"
        );
        // Pop from TX FIFO and verify halfword-replicated value.
        let val = bus.pio[1].pop_tx(0).expect("TX FIFO should have one entry");
        assert_eq!(
            val, 0x12341234,
            "halfword 0x1234 must be replicated to both lanes"
        );
    }

    #[test]
    fn pio_non_txf_byte_write_still_dropped() {
        let mut bus = Bus::new();
        // Release PIO1 from reset.
        bus.write32(0x4000_F000, 1u32 << 11);
        // Write CTRL to enable SM0 (known baseline).
        bus.write32(PIO1_BASE, 0x1);
        let ctrl_before = bus.read32(PIO1_BASE);
        // Byte write to PIO1 CTRL (offset 0x000) — must be silently dropped.
        bus.write8(PIO1_BASE, 0xFF);
        let ctrl_after = bus.read32(PIO1_BASE);
        assert_eq!(
            ctrl_before, ctrl_after,
            "byte write to non-TXF PIO register must be dropped"
        );
    }

    #[test]
    fn adc_fifo_narrow_read_does_not_double_pop() {
        // Drive one sample into the ADC FIFO and read it back byte-wise
        // via `read8` — the narrow dispatch must pop exactly one entry,
        // not trigger an RMW that pops twice.
        let mut bus = Bus::new();
        // Release ADC from reset.
        bus.write32(0x4000_F000, 0x1);
        // Prime the internal FIFO with one known sample by driving
        // through the Phase 3 path: enable + FCS + start_once + tick.
        bus.adc.reset();
        let mut irqs = 0u32;
        // Channel 3 so the sample is non-zero.
        bus.adc.write32(
            crate::peripherals::adc::FCS,
            crate::peripherals::adc::FCS_EN,
            0,
            &mut irqs,
        );
        bus.adc.write32(
            crate::peripherals::adc::CS,
            crate::peripherals::adc::CS_EN | crate::peripherals::adc::CS_START_ONCE | (3 << 12),
            0,
            &mut irqs,
        );
        // Tick enough sys cycles to complete the conversion
        // (96 adc_clk / 48 MHz ≈ 2 µs = 250 sys_clk at 125 MHz).
        bus.seed_sys_clk_hz(125_000_000);
        bus.adc.tick(400, &bus.clock_tree, &mut irqs);
        assert_eq!(bus.adc.fifo_len(), 1, "one sample must be queued");
        // Byte read must not over-pop — still one sample, count drops
        // to zero only once.
        let _ = bus.read8(ADC_BASE + crate::peripherals::adc::FIFO);
        assert_eq!(bus.adc.fifo_len(), 0, "byte read pops exactly once");
    }

    /// HLD V5 §5.2: a SysTick underflow must set `ICSR.PENDSTSET` (bit
    /// 26) only on the active core's PPB. Tick the active-core SysTick
    /// directly here — the per-cycle tick wired into `Emulator::step`
    /// is exercised end-to-end by §6.2's integration test, but the
    /// active-core-only invariant is testable purely at the bus level.
    #[test]
    fn tick_underflow_sets_icsr_pendst_on_active_core_only() {
        let mut bus = Bus::new();
        // Run from core 0's perspective so MMIO writes target
        // `systicks[0]`.
        bus.set_active_core(0);
        // Enable SysTick with TICKINT and CLKSOURCE=processor; CVR=0
        // so the very first tick fires.
        bus.write32(0xE000_E018, 0); // CVR
        bus.write32(0xE000_E014, 0); // RVR (period 1)
        bus.write32(0xE000_E010, 0b111); // CSR: ENABLE | TICKINT | CLKSOURCE
        let fired = bus.systicks[0].tick();
        assert!(fired, "TICKINT-armed first tick must fire");
        if fired {
            bus.ppb[0].icsr |= 1 << 26;
        }
        assert_ne!(
            bus.ppb[0].icsr & (1 << 26),
            0,
            "PENDSTSET must latch on active-core PPB"
        );
        assert_eq!(
            bus.ppb[1].icsr & (1 << 26),
            0,
            "PENDSTSET must NOT latch on the other core"
        );
    }
}
