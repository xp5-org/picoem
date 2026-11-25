//! Block engine against the interpreter: the same program on two emulators,
//! one with blocks off and one with blocks on or in check mode, must leave
//! identical registers, cycle counts, clock and SRAM.

use crate::{BlockMode, Config, Emulator};

const PROG: u32 = 0x2000_1000;
const DATA: u32 = 0x2000_2000;
const PARK: u32 = 0x2000_0F00;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

/// An emulator with core 1 halted, every vector pointing at a `b .` and
/// `prog` at [`PROG`].
fn setup(prog: &[u16], q: u32, mode: BlockMode, regs: &[u32; 16], data: &[u32]) -> Emulator {
    setup_jit(prog, q, mode, false, regs, data)
}

fn setup_jit(prog: &[u16], q: u32, mode: BlockMode, jit: bool, regs: &[u32; 16], data: &[u32]) -> Emulator {
    let mut emu = Emulator::new(Config::default());
    emu.step_quantum = q;
    emu.blocks.mode = mode;
    emu.blocks.jit = jit;
    for v in 0..48 {
        emu.bus.write32(0x2000_0000 + 4 * v, PARK | 1);
    }
    emu.bus.write16(PARK, 0xE7FE);
    emu.bus.ppb[0].vtor = 0x2000_0000;
    for (i, hw) in prog.iter().enumerate() {
        emu.bus.write16(PROG + 2 * i as u32, *hw);
    }
    for (i, w) in data.iter().enumerate() {
        emu.bus.write32(DATA + 4 * i as u32, *w);
    }
    let c = &mut emu.cores[0];
    c.regs.r = *regs;
    c.regs.msp = regs[13];
    c.regs.set_pc(PROG);
    emu
}

fn state(emu: &Emulator) -> (Vec<u32>, u64, u64, bool, u64) {
    let c = &emu.cores[0];
    let mut v = c.regs.r.to_vec();
    v.extend([c.regs.xpsr, c.regs.primask, c.regs.control, c.regs.msp, c.regs.psp]);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for off in (0..0x1_0000u32).step_by(4) {
        h = (h ^ emu.bus.memory.sram_read32(off) as u64).wrapping_mul(0x0100_0000_01b3);
    }
    (v, c.cycles, emu.clock.cycles, c.is_halted(), h)
}

fn run_pair(prog: &[u16], q: u32, regs: &[u32; 16], data: &[u32], cycles: u64, mode: BlockMode) -> (Emulator, Emulator) {
    run_pair_jit(prog, q, regs, data, cycles, mode, false)
}

fn run_pair_jit(prog: &[u16], q: u32, regs: &[u32; 16], data: &[u32], cycles: u64, mode: BlockMode, jit: bool) -> (Emulator, Emulator) {
    let mut a = setup(prog, q, BlockMode::Off, regs, data);
    let mut b = setup_jit(prog, q, mode, jit, regs, data);
    a.run(cycles).unwrap();
    b.run(cycles).unwrap();
    (a, b)
}

/// Random 16- and 32-bit encodings, faults and all, with registers that
/// mostly point into SRAM: the block engine and check mode agree with the
/// interpreter at every quantum size.
#[test]
fn blocks_match_interpreter_on_random_programs() {
    random_programs(false, 3000);
}

/// The same with hot blocks compiled to host code. The run is longer so
/// loops in the random code reach the compile threshold.
#[cfg(feature = "jit")]
#[test]
fn jit_matches_interpreter_on_random_programs() {
    random_programs(true, 20_000);
}

fn random_programs(jit: bool, cycles: u64) {
    // RP2040_BLOCK_FUZZ=N runs N cases instead of 600, with seed N.
    let n: usize = std::env::var("RP2040_BLOCK_FUZZ").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ n as u64 ^ jit as u64);
    let mut compiled = 0;
    let mut runs = 0;
    for case in 0..n {
        let mut prog = Vec::new();
        for _ in 0..48 {
            prog.push(rng.next() as u16);
        }
        // A backward branch to the start makes the program a loop, so its
        // blocks get hot.
        prog.push(if jit { 0xE7CE } else { 0xE7FE }); // b PROG, or b .
        let mut regs = [0u32; 16];
        for r in regs.iter_mut().take(13) {
            *r = if rng.next().is_multiple_of(4) { rng.next() } else { DATA + (rng.next() & 0x3FC) };
        }
        regs[13] = 0x2000_8000;
        regs[14] = PARK | 1;
        let data: Vec<u32> = (0..1024).map(|_| rng.next()).collect();
        let q = [1, 3, 7, 64, 512][case % 5];
        let mode = if case % 2 == 0 { BlockMode::On } else { BlockMode::Check };
        let (a, b) = run_pair_jit(&prog, q, &regs, &data, cycles, mode, jit);
        assert!(state(&a) == state(&b), "case {case} q {q}: {:?} vs {:?}", state(&a).0, state(&b).0);
        runs += b.blocks.stats.runs;
        compiled += b.blocks.stats.jit_runs;
    }
    assert!(runs > 10_000, "blocks ran only {runs} times");
    if jit {
        assert!(compiled > 10_000, "compiled blocks ran only {compiled} times");
    }
}

/// A store into a later instruction of the running block ends the block,
/// and the rewritten instruction runs.
#[test]
fn store_into_own_block_runs_new_code() {
    let prog = [
        0x2021, // movs r0, #0x21
        0x0200, // lsls r0, r0, #8
        0x3055, // adds r0, #0x55: r0 = movs r1, #0x55
        0x8018, // strh r0, [r3]
        0x46C0, // nop
        0x2111, // movs r1, #0x11, rewritten
        0x3201, // adds r2, #1
        0xE7FE, // b .
    ];
    let mut regs = [0u32; 16];
    regs[3] = PROG + 10;
    regs[13] = 0x2000_8000;
    for mode in [BlockMode::On, BlockMode::Check] {
        let (a, b) = run_pair(&prog, 512, &regs, &[], 2000, mode);
        assert_eq!(a.cores[0].regs.r[1], 0x55);
        assert!(state(&a) == state(&b));
    }
    // A second pass over code the block engine has already built.
    let mut prog2 = prog.to_vec();
    prog2[7] = 0xE7F2; // b back to the start
    let (a, b) = run_pair(&prog2, 512, &regs, &[], 2000, BlockMode::Check);
    assert!(state(&a) == state(&b));
}

/// An interrupt the program pends through the NVIC is taken at the next
/// block entry, as the interpreter takes it before the next instruction.
#[test]
fn nvic_pend_taken_at_block_entry() {
    // Vector 16 is IRQ 0.
    let handler = PROG + 0x40;
    let prog = [
        0x2001, // movs r0, #1
        0x6008, // str r0, [r1]   ISER
        0x6010, // str r0, [r2]   ISPR
        0x3501, // adds r5, #1
        0x3501, // adds r5, #1
        0xE7FC, // b back to the adds
    ];
    let mut full = prog.to_vec();
    full.resize(0x20, 0x46C0);
    full.extend([0x3601, 0x4770]); // adds r6, #1; bx lr
    let mut regs = [0u32; 16];
    regs[1] = 0xE000_E100;
    regs[2] = 0xE000_E200;
    regs[13] = 0x2000_8000;
    for q in [1, 5, 64] {
        let mut a = setup(&full, q, BlockMode::Off, &regs, &[]);
        let mut b = setup(&full, q, BlockMode::Check, &regs, &[]);
        for e in [&mut a, &mut b] {
            e.bus.write32(0x2000_0000 + 16 * 4, handler | 1);
            e.run(2000).unwrap();
        }
        assert_eq!(a.cores[0].regs.r[6], 1, "handler ran once");
        assert!(state(&a) == state(&b), "q {q}");
    }
}

/// A computing loop, then a poll on an SRAM flag that a SysTick handler
/// sets, with idle tracking on: loop heads are visited, the poll is
/// skipped, the computing head is marked busy. Blocks and compiled blocks
/// must leave the interpreter's state, cycle counts and skips.
#[test]
fn idle_tracking_matches_interpreter_with_blocks() {
    // The CLOCKS read makes a pass from `start` dirty while its snapshot
    // is live, so the visit at the computing loop's busy head must drop it.
    let mut prog = vec![
        0x683D, // ldr r5, [r7]       start: CLOCKS
        0x2000, // movs r0, #0
        0x3001, // adds r0, #1        loop
        0x6018, // str r0, [r3]
        0x28C8, // cmp r0, #200
        0xD1FB, // bne loop
        0x6811, // ldr r1, [r2]       poll
        0x2900, // cmp r1, #0
        0xD0FC, // beq poll
        0x2100, // movs r1, #0
        0x6011, // str r1, [r2]
        0xE7F3, // b start
    ];
    prog.resize(0x20, 0x46C0);
    prog.extend([0x2601, 0x6016, 0x4770]); // handler: movs r6, #1; str r6, [r2]; bx lr
    let mut regs = [0u32; 16];
    regs[2] = DATA;
    regs[3] = DATA + 0x100;
    regs[7] = 0x4000_8000;
    regs[13] = 0x2000_8000;
    let mut modes = vec![(BlockMode::Off, false), (BlockMode::On, false), (BlockMode::Check, false)];
    if cfg!(feature = "jit") {
        modes.extend([(BlockMode::On, true), (BlockMode::Check, true)]);
    }
    for q in [64, 512] {
        let mut runs = Vec::new();
        for &(mode, jit) in &modes {
            let mut e = setup_jit(&prog, q, mode, jit, &regs, &[0]);
            e.idle.enabled = true;
            e.bus.write32(0x2000_0000 + 15 * 4, (PROG + 0x40) | 1);
            let st = &mut e.bus.systicks[0];
            st.rvr = 2999;
            st.cvr = 0;
            st.csr = 0b111;
            e.run(400_000).unwrap();
            runs.push((state(&e), e.idle.skipped_cycles[0], e.cores[0].regs.r[0], e.blocks.stats.chained));
        }
        let (s0, skip0, ..) = &runs[0];
        assert!(*skip0 > 0, "q {q}: the poll was never skipped");
        for (i, (s, skip, _, _)) in runs.iter().enumerate() {
            assert!(s == s0 && skip == skip0, "q {q} mode {:?}: {:?} vs {:?}, skipped {skip} vs {skip0}", modes[i], s.0, s0.0);
        }
        if cfg!(feature = "jit") {
            assert!(runs[3].3 > 0, "q {q}: no chained blocks");
        }
    }
}

/// Watch mode, as PicoGUS runs: core 0 parks on a word nobody sets; core
/// 1 waits for TIMERAWL to pass a deadline, reading an SRAM word and
/// storing it back each pass, which restores itself. The TIMER read
/// ends core 1's park with each quantum, so its pass is tracked again every
/// quantum while core 0 stays parked, and compiled blocks call the tracking
/// hooks from their fast path. Every mode must leave the interpreter's
/// state of both cores, SRAM, the clock and the parked cycles. A second
/// run adds 1 to the word, which changes memory every pass while the
/// registers repeat: the pass must not park, and does if a store goes
/// unlogged.
#[test]
fn watch_tracking_matches_interpreter_with_blocks() {
    let mut poll = [
        0x683E, // 1000: ldr r6, [r7]      TIMERAWL
        0x6808, // 1002: ldr r0, [r1]
        0x3000, // 1004: adds r0, #0       or #1, the counting run
        0x6008, // 1006: str r0, [r1]
        0x2000, // 1008: movs r0, #0
        0x42A6, // 100A: cmp r6, r4
        0xD3F8, // 100C: bcc 1000          until the deadline
        0x3501, // 100E: adds r5, #1
        0xE7FD, // 1010: b 100E
    ];
    let wait = [
        0x6810, // 1100: ldr r0, [r2]      never set
        0x2800, // 1102: cmp r0, #0
        0xD0FC, // 1104: beq 1100
        0xE7FE, // 1106: b 1106
    ];
    let mut run = |q: u32, mode: BlockMode, jit: bool, stamp: bool| {
        poll[2] = 0x3000 | stamp as u16;
        let mut e = Emulator::new(Config::default());
        e.step_quantum = q;
        e.blocks.mode = mode;
        e.blocks.jit = jit;
        e.idle.enabled = true;
        e.idle.two_core = true;
        // TIMER out of reset, so its raw counters count.
        let r = e.bus.read32(0x4000_C000);
        e.bus.write32(0x4000_C000, r & !(1 << 21));
        for (base, prog) in [(0x2000_1000u32, &poll[..]), (0x2000_1100, &wait[..])] {
            for (i, hw) in prog.iter().enumerate() {
                e.bus.write16(base + 2 * i as u32, *hw);
            }
        }
        for (k, pc, sp) in [(0usize, 0x2000_1100u32, 0x2000_2000u32), (1, 0x2000_1000, 0x2000_3000)] {
            let c = &mut e.cores[k];
            c.regs.r[13] = sp;
            c.regs.msp = sp;
            c.regs.r[1] = 0x2000_0100;
            // Its own page: a parked core's words make their page slow.
            c.regs.r[2] = 0x2000_0800;
            c.regs.r[4] = 50_000;
            c.regs.r[5] = 0;
            c.regs.r[7] = 0x4005_4028;
            c.regs.set_pc(pc);
        }
        e.cores[1].wake();
        e.run(600_000).unwrap();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for off in (0..0x1_0000u32).step_by(4) {
            h = (h ^ e.bus.memory.sram_read32(off) as u64).wrapping_mul(0x0100_0000_01b3);
        }
        let cores: Vec<_> = e.cores.iter().map(|c| (c.regs.r, c.regs.xpsr, c.cycles)).collect();
        ((cores, h, e.clock.cycles, e.idle.parked_cycles), e.blocks.stats.jit_tracked)
    };
    let mut modes = vec![(BlockMode::On, false), (BlockMode::Check, false)];
    if cfg!(feature = "jit") {
        modes.extend([(BlockMode::On, true), (BlockMode::Check, true)]);
    }
    for q in [64, 512] {
        for stamp in [false, true] {
            let (s0, _) = run(q, BlockMode::Off, false, stamp);
            if stamp {
                assert!(s0.3[1] < 10_000, "q {q}: the counting pass parked {} cycles", s0.3[1]);
            } else {
                assert!(s0.3[0] > 100_000 && s0.3[1] > 100_000, "q {q}: parked only {:?} cycles", s0.3);
            }
            assert!(s0.0[1].0[5] > 1000, "q {q}: core 1 never passed its deadline");
            for &(mode, jit) in &modes {
                let (s, tracked) = run(q, mode, jit, stamp);
                assert!(s == s0, "q {q} stamp {stamp} mode {mode:?} jit {jit}: {:?} vs {:?}", (&s.0, s.2, s.3), (&s0.0, s0.2, s0.3));
                if jit && mode == BlockMode::On {
                    assert!(tracked > 0, "q {q}: compiled code never tracked an access");
                }
            }
        }
    }
}
