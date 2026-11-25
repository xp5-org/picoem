//! Block engine: runs straight-line runs of pre-decoded instructions
//! without the per-instruction bookkeeping of [`CortexM0Plus::step`].
//! Stage 1 of `wrk_docs/2026.11.25 - HLD - RP2040 M0+ JIT V1.md`.
//!
//! A block starts at an entry PC and ends after a branch, a write to PC,
//! any instruction the interpreter must see alone, 32 instructions, or the
//! end of a 1 KB page. Each instruction is lowered to an [`Op`]: the common
//! ALU, load, store and branch encodings get their own kind, everything else
//! calls the interpreter's executor.
//!
//! Exactness rules, each matching what `run_single` would do instruction by
//! instruction:
//!
//! - A block stops after the instruction that reaches the cycle budget.
//! - Exceptions are taken only at block entry; every instruction that can
//!   make one pending for this core ends its block.
//! - A block ends after a store outside SRAM, after a store that queues a
//!   decode-cache invalidation, after an access that sets the other core's
//!   park conflict, and after a fault. MMIO reads do not end it: nothing the
//!   scheduler checks between instructions changes on a read.
//! - SRAM accesses take the fast path only while idle tracking, the MMIO
//!   trace and the bank contention window are all off, and only on pages
//!   that hold no code and no parked core's watched words.
//!
//! In check mode every block is re-run on the interpreter against a
//! [`Sandbox`] bus that holds the memory as it was at block entry and
//! replays the block's MMIO reads; registers, flags, cycles, SRAM stores and
//! MMIO stores must agree.

use std::collections::HashMap;

use super::bus_trait::CoreBus;
use super::decode::is_wide;
use super::execute::{add_with_carry, is_aligned, sign_extend};
use super::{CortexM0Plus, Fault, Registers};
use crate::bus::Bus;
use crate::bus::ppb::Ppb;
use crate::core::Nvic;
use crate::memory::SRAM_SIZE;

#[cfg(feature = "jit")]
mod jit;

/// Runs of a block before it is compiled.
#[cfg(feature = "jit")]
const JIT_HOT: u32 = 64;

/// Most instructions in one block.
pub const BLOCK_MAX_OPS: usize = 32;
/// Direct-mapped block slots per core.
const SLOTS: usize = 16384;
/// Largest byte span of a block, from its entry to the end of its last
/// instruction; forward branches it follows count the bytes they skip.
const BLOCK_MAX_BYTES: u32 = 256;

/// Whether `run_single` runs blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlockMode {
    /// Interpreter only, the reference path.
    #[default]
    Off,
    /// Blocks where they apply.
    On,
    /// Blocks, each re-run on the interpreter in a sandbox and compared;
    /// the first mismatch panics with a report.
    Check,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum K {
    LslI,
    LsrI,
    AsrI,
    AddR,
    SubR,
    AddI,
    SubI,
    MovI,
    CmpI,
    And,
    Eor,
    LslR,
    LsrR,
    AsrR,
    Adc,
    Sbc,
    Ror,
    Tst,
    Neg,
    CmpR,
    Cmn,
    Orr,
    Mul,
    Bic,
    Mvn,
    AddHi,
    CmpHi,
    MovHi,
    /// `d = imm`, no flags: ADR and MOV from PC.
    MovK,
    /// `d = mem32[imm]`: LDR literal.
    LdrK,
    StrR,
    StrhR,
    StrbR,
    LdrsbR,
    LdrR,
    LdrhR,
    LdrbR,
    LdrshR,
    /// `[n + imm]`; `n` may be SP.
    StrI,
    LdrI,
    StrbI,
    LdrbI,
    StrhI,
    LdrhI,
    /// `d = sp + imm`.
    AddSpK,
    AddSp,
    SubSp,
    Sxth,
    Sxtb,
    Uxth,
    Uxtb,
    Rev,
    Rev16,
    Revsh,
    Push,
    Pop,
    Stm,
    Ldm,
    Nop,
    /// Runs the interpreter's executor; never writes PC or touches memory.
    Interp,
    BCond,
    B,
    Bl,
    /// Runs the interpreter's executor and ends the block.
    InterpEnd,
}

impl K {
    #[inline(always)]
    fn ends_block(self) -> bool {
        matches!(self, K::BCond | K::B | K::Bl | K::InterpEnd)
    }
}

/// One lowered instruction. `off` is its byte offset from the block's
/// entry PC. For the interpreter kinds `imm` holds `hw1` and `m` is 1 for a
/// wide encoding.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Op {
    pub k: K,
    pub d: u8,
    pub n: u8,
    pub m: u8,
    pub off: u16,
    pub hw0: u16,
    pub imm: u32,
}

impl Op {
    #[inline(always)]
    fn len(&self) -> u32 {
        if self.k == K::Bl || (matches!(self.k, K::Interp | K::InterpEnd) && self.m != 0) { 4 } else { 2 }
    }
}

pub(crate) struct Block {
    pub pc: u32,
    pub end: u32,
    pub ops: Box<[Op]>,
    /// Runs so far, counted until the block is compiled.
    #[cfg(feature = "jit")]
    runs: u32,
    #[cfg(feature = "jit")]
    code: Option<jit::JitFn>,
    /// The block's end leaves nothing for the scheduler to do: a branch, a
    /// return or a plain fall-through, so the next block may follow
    /// directly; see [`CortexM0Plus::run_chain`].
    #[cfg(feature = "jit")]
    chain: bool,
    /// Compiled code tests the budget only after its checkpoints, see
    /// [`plain_cycles`]; it may start only with more budget left than this,
    /// the cycles of the plain ops before the first checkpoint.
    #[cfg(feature = "jit")]
    pub entry_need: u64,
}

/// The cycles of an op that never ends a compiled block early on the
/// budget or the exit flag, or `None` for a checkpoint: a memory access,
/// whose slow path may ask for an exit and which may fault, and an
/// interpreter op, whose cycles vary. Compiled code tests the exit flag
/// and the budget only after checkpoints, so a run of plain ops between
/// two checkpoints goes unchecked. A branch inside a block counts its
/// taken cycles; when it leaves the block's path the block exits there.
#[cfg(feature = "jit")]
pub(crate) fn plain_cycles(op: &Op) -> Option<u64> {
    match op.k {
        K::LdrK
        | K::StrR
        | K::StrhR
        | K::StrbR
        | K::LdrsbR
        | K::LdrR
        | K::LdrhR
        | K::LdrbR
        | K::LdrshR
        | K::StrI
        | K::LdrI
        | K::StrbI
        | K::LdrbI
        | K::StrhI
        | K::LdrhI
        | K::Push
        | K::Pop
        | K::Stm
        | K::Ldm
        | K::Interp
        | K::InterpEnd => None,
        K::BCond | K::B => Some(3),
        K::Bl => Some(4),
        _ => Some(1),
    }
}

/// A run of `b` that stopped after `done` ops left it through a branch
/// before its last op. Branches are plain, so no budget or exit-flag stop
/// ends a run on one.
#[cfg(feature = "jit")]
fn side_exit(b: &Block, done: usize) -> bool {
    done > 0 && done < b.ops.len() && b.ops[done - 1].k == K::BCond
}

/// For each op of `ops`, the cycles of the plain ops that follow it up to
/// the next checkpoint, leaving out the last op, whose budget test is the
/// caller's; index 0 of the result is for the stretch before op 0.
#[cfg(feature = "jit")]
pub(crate) fn plain_runs(ops: &[Op]) -> Vec<u64> {
    let n = ops.len();
    let mut out = vec![0u64; n + 1];
    let mut run = 0u64;
    for i in (0..n).rev() {
        out[i + 1] = run;
        if i + 1 == n {
            continue;
        }
        match plain_cycles(&ops[i]) {
            Some(c) => run += c,
            None => run = 0,
        }
    }
    out[0] = run;
    out
}

impl Block {
    fn new(pc: u32, end: u32, ops: Box<[Op]>) -> Self {
        #[cfg(feature = "jit")]
        let chain = ops.last().is_some_and(|op| match op.k {
            // BX and BLX, POP with PC, MOV and ADD to PC.
            K::InterpEnd => op.m == 0 && (op.hw0 >> 8 == 0x47 || op.hw0 >> 8 == 0xBD || (op.hw0 >> 10 == 0x11 && op.hw0 & 0x87 == 0x87 && (op.hw0 >> 8) & 3 != 3)),
            _ => true,
        });
        #[cfg(feature = "jit")]
        let entry_need = plain_runs(&ops)[0];
        Self {
            pc,
            end,
            ops,
            #[cfg(feature = "jit")]
            runs: 0,
            #[cfg(feature = "jit")]
            code: None,
            #[cfg(feature = "jit")]
            chain,
            #[cfg(feature = "jit")]
            entry_need,
        }
    }

    fn empty() -> Self {
        Self::new(u32::MAX, 0, Box::new([]))
    }
}

/// Counters for one emulator; see [`Blocks::stats`].
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockStats {
    pub built: u64,
    pub runs: u64,
    pub ops: u64,
    /// Steps run by the interpreter inside `run_single` because no block
    /// applied: an exception entry, or a PC outside ROM and SRAM.
    pub interp_steps: u64,
    pub evicted: u64,
    /// Instructions run with the SRAM fast path off: idle tracking, MMIO
    /// trace or the contention window.
    pub slow_ops: u64,
    /// Blocks ended by a store outside SRAM.
    pub io_exits: u64,
    /// Slow-path SRAM stores to a page holding code: each queues a
    /// decode-cache invalidation and ends its block.
    pub code_page_stores: u64,
    pub checked: u64,
    /// Blocks check mode could not compare: they faulted.
    pub unchecked: u64,
    pub mismatches: u64,
    /// Blocks compiled to host code.
    pub compiled: u64,
    /// Block runs on host code.
    pub jit_runs: u64,
    /// Times the code memory filled and was rebuilt.
    pub jit_resets: u64,
    /// Runs of [`CortexM0Plus::run_chain`]; `jit_runs` over this is the
    /// mean chain length.
    pub chains: u64,
    /// Blocks run inside chains.
    pub chained: u64,
    /// Why a chain did not start or stopped: fast path off, block not
    /// compiled, exception pending, slow-path exit or fault, budget, block
    /// end that does not chain, loop head that needs its visit.
    pub chain_stops: [u64; 7],
    /// Slow-path accesses from compiled code: loads and stores with the
    /// fast path off, to SRAM pages it skips, and outside SRAM.
    pub jit_slow: [u64; 6],
    /// Compiled-code fast accesses that called the idle-tracking hooks.
    pub jit_tracked: u64,
}

/// What `run_single` knows about the loop-head visit it would make after a
/// backward branch; see [`CortexM0Plus::run_chain`].
#[cfg(feature = "jit")]
pub(crate) struct ChainCtl<'a> {
    /// Heads are visited: the core is eligible for idle tracking.
    pub track: bool,
    /// The failure limit is reached, so a visit without a snapshot is a
    /// no-op.
    pub fails_maxed: bool,
    /// The live snapshot's head PC, SP and LR. A visit to another head, or
    /// with another SP or LR, is a no-op while the pass is clean; any other
    /// visit counts.
    pub snap: Option<(u32, u32, u32)>,
    /// Busy heads and the clock until which each stays busy.
    pub busy: &'a [(u32, u64)],
    /// The master clock at the chain's start.
    pub clock: u64,
}

/// What one block run did.
pub(crate) struct Exit {
    pub cycles: u64,
    /// PC of the last instruction run, for the loop-head test.
    pub last_pc: u32,
    /// Instructions the block ran; 0 for an exception entry or a step.
    pub ops: u32,
    /// Cycles of the last instruction, fault entry included; the rest of
    /// `cycles` came before it.
    pub last_cycles: u64,
    /// The SRAM fast path was on.
    pub fast: bool,
    pub io: u32,
    pub code_stores: u32,
}

/// Per-emulator block state: one cache per core, mode, counters and the
/// check-mode scratch.
pub struct Blocks {
    pub mode: BlockMode,
    /// Compile hot blocks to host code. Needs the `jit` feature, and the
    /// host Cranelift supports; otherwise blocks stay on the micro-op loop.
    pub jit: bool,
    pub stats: BlockStats,
    /// First mismatches found in check mode, as printable reports.
    pub reports: Vec<String>,
    cache: [Vec<Block>; 2],
    scratch: Option<Box<CortexM0Plus>>,
    log: CheckLog,
    #[cfg(feature = "jit")]
    backend: Option<Box<jit::Jit>>,
    /// The host has no code generator.
    #[cfg(feature = "jit")]
    no_backend: bool,
}

impl Default for Blocks {
    fn default() -> Self {
        Self {
            mode: BlockMode::Off,
            jit: false,
            stats: BlockStats::default(),
            reports: Vec::new(),
            cache: [Vec::new(), Vec::new()],
            scratch: None,
            log: CheckLog::default(),
            #[cfg(feature = "jit")]
            backend: None,
            #[cfg(feature = "jit")]
            no_backend: false,
        }
    }
}

#[derive(Default)]
pub(crate) struct CheckLog {
    /// SRAM stores by the block: offset, size, old value.
    sram_old: Vec<(u32, u8, u32)>,
    /// Slow-path accesses outside SRAM in order: write, addr, size, value.
    io: Vec<(bool, u32, u8, u32)>,
}

#[inline(always)]
fn slot(pc: u32) -> usize {
    ((pc >> 1) as usize) & (SLOTS - 1)
}

impl Blocks {
    /// Drops every block of both cores.
    pub fn clear(&mut self) {
        for c in &mut self.cache {
            c.clear();
        }
    }

    /// Evicts the blocks of both cores that cover any halfword in `addrs`,
    /// the addresses a store queued for decode-cache invalidation.
    pub(crate) fn invalidate(&mut self, addrs: &[u32]) {
        for c in &mut self.cache {
            if c.is_empty() {
                continue;
            }
            for &a in addrs {
                let h = a & !1;
                let lo = h.saturating_sub(BLOCK_MAX_BYTES - 2);
                let mut pc = lo;
                while pc <= h {
                    let b = &mut c[slot(pc)];
                    if b.pc == pc && h < b.end {
                        *b = Block::empty();
                        self.stats.evicted += 1;
                    }
                    pc += 2;
                }
            }
        }
    }

    fn lookup_or_build(&mut self, core: usize, bus: &mut Bus, pc: u32) -> Option<usize> {
        let c = &mut self.cache[core];
        if c.is_empty() {
            c.resize_with(SLOTS, Block::empty);
        }
        let i = slot(pc);
        if c[i].pc == pc {
            return Some(i);
        }
        let b = build(bus, pc)?;
        self.stats.built += 1;
        c[i] = b;
        Some(i)
    }

    /// Counts a run of block `i` and compiles it once hot.
    #[cfg(feature = "jit")]
    fn heat(&mut self, core: usize, i: usize) {
        let b = &mut self.cache[core][i];
        if b.code.is_some() || b.runs > JIT_HOT || self.no_backend {
            return;
        }
        b.runs += 1;
        if b.runs < JIT_HOT {
            return;
        }
        if self.backend.as_ref().is_some_and(|j| j.compiled >= jit::JIT_MAX_BLOCKS) {
            // Code memory is freed only as a whole: drop every compiled
            // block first, then the module.
            for c in &mut self.cache {
                for b in c.iter_mut() {
                    b.code = None;
                    b.runs = 0;
                }
            }
            self.backend = None;
            self.stats.jit_resets += 1;
        }
        if self.backend.is_none() {
            match jit::Jit::new() {
                Some(j) => self.backend = Some(Box::new(j)),
                None => {
                    self.no_backend = true;
                    return;
                }
            }
        }
        let jit = self.backend.as_mut().expect("backend");
        let b = &mut self.cache[core][i];
        b.code = jit.compile(b, core == 0);
        if b.code.is_some() {
            self.stats.compiled += 1;
        } else {
            // Never retried.
            b.runs = JIT_HOT + 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

fn build(bus: &mut Bus, pc: u32) -> Option<Block> {
    bus.peek_code16(pc)?;
    let page_end = (pc | 0x3FF).wrapping_add(1);
    let mut ops = Vec::with_capacity(BLOCK_MAX_OPS);
    let mut cur = pc;
    while ops.len() < BLOCK_MAX_OPS {
        let Some(hw0) = bus.peek_code16(cur) else { break };
        let wide = is_wide(hw0);
        let len = if wide { 4 } else { 2 };
        if cur.wrapping_add(len) > page_end || page_end == 0 || cur.wrapping_add(len) - pc > BLOCK_MAX_BYTES {
            break;
        }
        let hw1 = if wide {
            match bus.peek_code16(cur + 2) {
                Some(h) => h,
                None => break,
            }
        } else {
            0
        };
        let op = lower(hw0, hw1, cur, (cur - pc) as u16);
        ops.push(op);
        cur += len;
        if op.k.ends_block() {
            match follow(bus, &op, cur) {
                Some(next) => cur = next,
                None => break,
            }
        }
    }
    if ops.is_empty() {
        return None;
    }
    bus.set_code_page(pc);
    bus.set_code_page(cur - 1);
    Some(Block::new(pc, cur, ops.into_boxed_slice()))
}

/// Where a block goes on after branch `op`, whose next instruction is at
/// `next`, or `None` when the branch ends it. A block follows only forward
/// branches, so it never holds a loop head's backward branch except as its
/// last op; `build` keeps the path inside the block's page and byte span.
/// B goes on at its target. BCond goes on at its fall-through and leaves
/// the block when taken, except before an unconditional B it skips, the
/// way GCC jumps to far code: then the taken edge is the one followed.
fn follow(bus: &mut Bus, op: &Op, next: u32) -> Option<u32> {
    match op.k {
        K::B if op.imm >= next => Some(op.imm),
        K::BCond if op.imm > next => {
            let skips_b = op.imm == next + 2 && bus.peek_code16(next).is_some_and(|h| h >> 11 == 0b11100);
            Some(if skips_b { op.imm } else { next })
        }
        _ => None,
    }
}

/// Lowers one instruction at `pc`; `off` is its offset in the block.
pub(crate) fn lower(hw0: u16, hw1: u16, pc: u32, off: u16) -> Op {
    let mk = |k: K, d: u16, n: u16, m: u16, imm: u32| Op { k, d: d as u8, n: n as u8, m: m as u8, off, hw0, imm };
    let interp = |end: bool, wide: bool| Op {
        k: if end { K::InterpEnd } else { K::Interp },
        d: 0,
        n: 0,
        m: wide as u8,
        off,
        hw0,
        imm: hw1 as u32,
    };
    if is_wide(hw0) {
        if hw1 & 0xD000 == 0xD000 {
            let s = ((hw0 >> 10) & 1) as u32;
            let j1 = ((hw1 >> 13) & 1) as u32;
            let j2 = ((hw1 >> 11) & 1) as u32;
            let i1 = (j1 ^ s) ^ 1;
            let i2 = (j2 ^ s) ^ 1;
            let imm25 = (s << 24) | (i1 << 23) | (i2 << 22) | (((hw0 & 0x3FF) as u32) << 12) | (((hw1 & 0x7FF) as u32) << 1);
            let target = pc.wrapping_add(4).wrapping_add(sign_extend(imm25, 25)) & !1;
            return mk(K::Bl, 0, 0, 0, target);
        }
        return interp(true, true);
    }
    let o = hw0;
    let lo3 = o & 7;
    let mid3 = (o >> 3) & 7;
    let hi3 = (o >> 6) & 7;
    let r8 = (o >> 8) & 7;
    let imm8 = (o & 0xFF) as u32;
    let imm5 = ((o >> 6) & 0x1F) as u32;
    let pc4 = pc.wrapping_add(4);
    match o >> 11 {
        0b00000 => mk(K::LslI, lo3, 0, mid3, imm5),
        0b00001 => mk(K::LsrI, lo3, 0, mid3, imm5),
        0b00010 => mk(K::AsrI, lo3, 0, mid3, imm5),
        0b00011 => match (o >> 9) & 3 {
            0 => mk(K::AddR, lo3, mid3, hi3, 0),
            1 => mk(K::SubR, lo3, mid3, hi3, 0),
            2 => mk(K::AddI, lo3, mid3, 0, hi3 as u32),
            _ => mk(K::SubI, lo3, mid3, 0, hi3 as u32),
        },
        0b00100 => mk(K::MovI, r8, 0, 0, imm8),
        0b00101 => mk(K::CmpI, 0, r8, 0, imm8),
        0b00110 => mk(K::AddI, r8, r8, 0, imm8),
        0b00111 => mk(K::SubI, r8, r8, 0, imm8),
        0b01000 if o & (1 << 10) == 0 => {
            const DP: [K; 16] = [
                K::And,
                K::Eor,
                K::LslR,
                K::LsrR,
                K::AsrR,
                K::Adc,
                K::Sbc,
                K::Ror,
                K::Tst,
                K::Neg,
                K::CmpR,
                K::Cmn,
                K::Orr,
                K::Mul,
                K::Bic,
                K::Mvn,
            ];
            mk(DP[((o >> 6) & 0xF) as usize], lo3, lo3, mid3, 0)
        }
        0b01000 => {
            let dn = ((o >> 4) & 8) | lo3;
            let rm = (o >> 3) & 0xF;
            match (o >> 8) & 3 {
                0 if dn == 15 => interp(true, false),
                0 if rm == 15 => interp(false, false),
                0 => mk(K::AddHi, dn, dn, rm, 0),
                1 if dn == 15 || rm == 15 => interp(false, false),
                1 => mk(K::CmpHi, 0, dn, rm, 0),
                2 if dn == 15 => interp(true, false),
                2 if rm == 15 => mk(K::MovK, dn, 0, 0, pc4),
                2 => mk(K::MovHi, dn, 0, rm, 0),
                _ => interp(true, false),
            }
        }
        0b01001 => mk(K::LdrK, r8, 0, 0, (pc4 & !3).wrapping_add(imm8 << 2)),
        0b01010 | 0b01011 => {
            const LS: [K; 8] = [K::StrR, K::StrhR, K::StrbR, K::LdrsbR, K::LdrR, K::LdrhR, K::LdrbR, K::LdrshR];
            mk(LS[((o >> 9) & 7) as usize], lo3, mid3, hi3, 0)
        }
        0b01100 => mk(K::StrI, lo3, mid3, 0, imm5 << 2),
        0b01101 => mk(K::LdrI, lo3, mid3, 0, imm5 << 2),
        0b01110 => mk(K::StrbI, lo3, mid3, 0, imm5),
        0b01111 => mk(K::LdrbI, lo3, mid3, 0, imm5),
        0b10000 => mk(K::StrhI, lo3, mid3, 0, imm5 << 1),
        0b10001 => mk(K::LdrhI, lo3, mid3, 0, imm5 << 1),
        0b10010 => mk(K::StrI, r8, 13, 0, imm8 << 2),
        0b10011 => mk(K::LdrI, r8, 13, 0, imm8 << 2),
        0b10100 => mk(K::MovK, r8, 0, 0, (pc4 & !3).wrapping_add(imm8 << 2)),
        0b10101 => mk(K::AddSpK, r8, 0, 0, imm8 << 2),
        0b10110 | 0b10111 => match (o >> 8) & 0xF {
            0b0000 if o & 0x80 == 0 => mk(K::AddSp, 0, 0, 0, ((o & 0x7F) as u32) << 2),
            0b0000 => mk(K::SubSp, 0, 0, 0, ((o & 0x7F) as u32) << 2),
            0b0010 => {
                let k = [K::Sxth, K::Sxtb, K::Uxth, K::Uxtb][((o >> 6) & 3) as usize];
                mk(k, lo3, 0, mid3, 0)
            }
            0b0100 | 0b0101 => mk(K::Push, 0, 0, 0, imm8 | if o & 0x100 != 0 { 1 << 14 } else { 0 }),
            0b1010 => match (o >> 6) & 3 {
                0 => mk(K::Rev, lo3, 0, mid3, 0),
                1 => mk(K::Rev16, lo3, 0, mid3, 0),
                3 => mk(K::Revsh, lo3, 0, mid3, 0),
                _ => interp(true, false),
            },
            0b1100 | 0b1101 if o & 0x100 == 0 => mk(K::Pop, 0, 0, 0, imm8),
            0b1111 if o & 0xF == 0 && !matches!((o >> 4) & 0xF, 2..=4) => mk(K::Nop, 0, 0, 0, 0),
            _ => interp(true, false),
        },
        0b11000 => mk(K::Stm, 0, r8, 0, imm8),
        0b11001 => mk(K::Ldm, 0, r8, 0, imm8),
        0b11010 | 0b11011 => {
            let cond = (o >> 8) & 0xF;
            if cond >= 0xE {
                interp(true, false)
            } else {
                mk(K::BCond, 0, 0, cond, pc4.wrapping_add(sign_extend(imm8 << 1, 9)))
            }
        }
        0b11100 => mk(K::B, 0, 0, 0, pc4.wrapping_add(sign_extend(((o & 0x7FF) as u32) << 1, 12))),
        _ => interp(true, false),
    }
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

struct Ctx {
    /// Compiled-code slow-path accesses: loads and stores with the fast
    /// path off, to SRAM pages it skips, and outside SRAM.
    #[cfg(feature = "jit")]
    slow_kind: [u64; 6],
    /// Compiled-code fast accesses that called the idle-tracking hooks.
    #[cfg(feature = "jit")]
    tracked: u64,
    fast: bool,
    exit: bool,
    other: usize,
    io: u32,
    code_stores: u32,
}

/// Memory access helpers. The fast path reads and writes SRAM directly; the
/// slow path is the [`CoreBus`] method the interpreter calls.
macro_rules! mem_ops {
    ($($ld:ident, $st:ident, $size:expr, $ty:ty, $rd:ident, $wr:ident, $crd:ident, $cwr:ident;)*) => {
        impl CortexM0Plus {
            $(
            #[inline(always)]
            fn $ld<const CHECK: bool>(bus: &mut Bus, cx: &mut Ctx, log: &mut CheckLog, addr: u32, ipc: u32) -> u32 {
                if cx.fast {
                    if let Some(o) = bus.jit_sram(addr, $size, false) {
                        return bus.memory.$rd(o) as u32;
                    }
                }
                Self::slow::<CHECK>(bus, cx, log, addr, $size, None, ipc, |b, a, _| CoreBus::$crd(b, a) as u32)
            }

            #[inline(always)]
            fn $st<const CHECK: bool>(bus: &mut Bus, cx: &mut Ctx, log: &mut CheckLog, addr: u32, val: u32, ipc: u32) {
                if cx.fast {
                    if let Some(o) = bus.jit_sram(addr, $size, true) {
                        if CHECK {
                            log.sram_old.push((o, $size, bus.memory.$rd(o) as u32));
                        }
                        bus.memory.$wr(o, val as $ty);
                        return;
                    }
                }
                Self::slow::<CHECK>(bus, cx, log, addr, $size, Some(val), ipc, |b, a, v| {
                    CoreBus::$cwr(b, a, v as $ty);
                    0
                });
            }
            )*
        }
    };
}

mem_ops! {
    ld8, st8, 1, u8, sram_read8, sram_write8, read8, write8;
    ld16, st16, 2, u16, sram_read16, sram_write16, read16, write16;
    ld32, st32, 4, u32, sram_read32, sram_write32, read32, write32;
}

impl CortexM0Plus {
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn slow<const CHECK: bool>(
        bus: &mut Bus,
        cx: &mut Ctx,
        log: &mut CheckLog,
        addr: u32,
        size: u8,
        val: Option<u32>,
        ipc: u32,
        f: impl FnOnce(&mut Bus, u32, u32) -> u32,
    ) -> u32 {
        bus.set_active_pc(ipc);
        let inv = bus.pending_cache_invalidations.len();
        let hit = bus.park[cx.other].hit;
        let sram = addr >> 24 == 0x20 && ((addr & 0x00FF_FFFF) + size as u32) as usize <= SRAM_SIZE;
        if CHECK && sram && val.is_some() {
            let off = addr & 0x00FF_FFFF;
            let old = match size {
                1 => bus.memory.sram_read8(off) as u32,
                2 => bus.memory.sram_read16(off) as u32,
                _ => bus.memory.sram_read32(off),
            };
            log.sram_old.push((off, size, old));
        }
        let r = f(bus, addr, val.unwrap_or(0));
        if CHECK && !sram {
            let mask = if size == 4 { u32::MAX } else { (1u32 << (8 * size)) - 1 };
            log.io.push((val.is_some(), addr, size, val.map_or(r, |v| v & mask)));
        }
        // An MMIO read changes nothing the scheduler checks between
        // instructions: IRQ levels, PIO and timers move at quantum ends and
        // FIFO events come from writes. A parked core's conflict is caught
        // by its hit flag.
        let code = bus.pending_cache_invalidations.len() != inv;
        let io_store = !sram && val.is_some();
        cx.io += io_store as u32;
        cx.code_stores += code as u32;
        if io_store || code || bus.bus_fault() || bus.park[cx.other].hit != hit {
            cx.exit = true;
        }
        r
    }

    /// Runs one step of `run_single`'s work for this core: an exception
    /// entry, one block, or one interpreter step where no block applies.
    /// Stops after the instruction that brings the cycles run to `budget`.
    pub(crate) fn run_block(&mut self, bus: &mut Bus, blocks: &mut Blocks, budget: u64) -> Exit {
        let pc = self.regs.pc();
        if self.halted {
            return Exit { cycles: 0, last_pc: pc, ops: 0, last_cycles: 0, fast: true, io: 0, code_stores: 0 };
        }
        let core = self.core_id as usize;
        if bus.ppb[core].icsr & (1 << 28 | 1 << 26) != 0 || bus.nvics[core].pending_and_enabled() != 0 {
            let e = self.try_take_any_pending_exception(bus);
            if e != 0 {
                self.cycles = self.cycles.wrapping_add(e as u64);
                blocks.stats.interp_steps += 1;
                return Exit { cycles: e as u64, last_pc: pc, ops: 0, last_cycles: e as u64, fast: true, io: 0, code_stores: 0 };
            }
        }
        let Some(i) = blocks.lookup_or_build(core, bus, pc) else {
            blocks.stats.interp_steps += 1;
            let c = self.step(bus) as u64;
            return Exit { cycles: c, last_pc: pc, ops: 0, last_cycles: c, fast: true, io: 0, code_stores: 0 };
        };
        #[cfg(feature = "jit")]
        if blocks.jit {
            blocks.heat(core, i);
        }
        // The block is taken out of its slot while it runs, so stores that
        // evict blocks cannot touch it; the drain runs after the block.
        let blk = std::mem::replace(&mut blocks.cache[core][i], Block::empty());
        #[cfg(feature = "jit")]
        let code = blk.code.filter(|_| budget > blk.entry_need);
        #[cfg(not(feature = "jit"))]
        let code: Option<()> = None;
        blocks.stats.jit_runs += code.is_some() as u64;
        let mut exit = if blocks.mode == BlockMode::Check {
            let pre = self.regs;
            blocks.log.sram_old.clear();
            blocks.log.io.clear();
            let (exit, upto) = match code {
                #[cfg(feature = "jit")]
                Some(f) => self.run_jit::<true>(bus, f, &blk, budget, &mut blocks.log, &mut blocks.stats),
                _ => self.exec::<true>(bus, &blk, budget, &mut blocks.log),
            };
            self.check_block(bus, blocks, &blk, pre, upto);
            exit
        } else {
            match code {
                #[cfg(feature = "jit")]
                Some(f) => self.run_jit::<false>(bus, f, &blk, budget, &mut blocks.log, &mut blocks.stats).0,
                _ => self.exec::<false>(bus, &blk, budget, &mut blocks.log).0,
            }
        };
        let total = self.retire_faults(bus, exit.cycles as u32) as u64;
        exit.last_cycles += total - exit.cycles;
        exit.cycles = total;
        self.cycles = self.cycles.wrapping_add(total);
        blocks.stats.runs += 1;
        blocks.stats.ops += exit.ops as u64;
        if !exit.fast {
            blocks.stats.slow_ops += exit.ops as u64;
        }
        blocks.stats.io_exits += (exit.io != 0) as u64;
        blocks.stats.code_page_stores += exit.code_stores as u64;
        let c = &mut blocks.cache[core][i];
        if c.pc == u32::MAX {
            *c = blk;
        }
        exit
    }

    /// Runs compiled blocks one after another while `run_single` would do
    /// nothing between them, and returns them as one exit; `None` when the
    /// first block cannot run this way, for [`Self::run_block`] to take.
    ///
    /// Between two blocks `run_single` would take a pending exception, visit
    /// a loop head after a backward branch, drain invalidations, wake core 1
    /// or stop for a halt, a WFE, a park or the budget. A block that exits
    /// through the slow path, faults, stops on the budget or ends in an
    /// instruction other than a branch or return ends the chain, and so does
    /// a backward branch to a head whose visit would not be a no-op.
    #[cfg(feature = "jit")]
    #[inline(never)]
    pub(crate) fn run_chain(&mut self, bus: &mut Bus, blocks: &mut Blocks, budget: u64, ctl: &ChainCtl<'_>) -> Option<Exit> {
        let core = self.core_id as usize;
        // With idle tracking live fast accesses call the tracking hooks
        // themselves; with `ctl.snap` set the chain ends at the backward
        // branches whose visit counts.
        let fast = bus.jit_fast();
        if blocks.mode != BlockMode::On || self.halted || bus.mmio_trace_enabled || bus.contention_check_active || blocks.cache[core].is_empty() {
            blocks.stats.chain_stops[0] += 1;
            return None;
        }
        let mut cx = Ctx {
            #[cfg(feature = "jit")]
            slow_kind: [0; 6],
            #[cfg(feature = "jit")]
            tracked: 0,
            fast, exit: false, other: 1 - core, io: 0, code_stores: 0 };
        let mut upto = None;
        let me: *mut Self = self;
        let busp: *mut Bus = bus;
        // SAFETY: as in `run_jit`; between calls only plain fields are read.
        let mut st = unsafe {
            jit::JitState {
                regs: &raw mut (*me).regs,
                sram: (*busp).memory.sram_ptr(),
                pages: (*busp).jit_pages.as_ptr(),
                bank: &raw mut (*busp).core0_bank_touched,
                budget,
                acc: 0,
                last_cycles: 0,
                last_pc: 0,
                ops: 0,
                core: me,
                bus: busp,
                cx: &raw mut cx,
                log: &raw mut blocks.log,
                blk: std::ptr::null(),
                upto: &raw mut upto,
                check: false,
                timer_ok: 0,
                timer: [0; 2],
                track: 0,
            }
        };
        st.set_mode(bus, true);
        let (mut total, mut runs, mut ops, mut last_pc, mut last_cycles) = (0u64, 0u64, 0u64, 0u32, 0u64);
        loop {
            // SAFETY: plain reads of the bus and core through the pointers
            // the state holds; no reference to either is live.
            let (b, pending) = unsafe {
                let bus = &*busp;
                (&blocks.cache[core][slot((*me).regs.r[15])], bus.ppb[core].icsr & (1 << 28 | 1 << 26) != 0 || bus.nvics[core].pending_and_enabled() != 0)
            };
            let pc = unsafe { (*me).regs.r[15] };
            let f = match b.code {
                Some(f) if b.pc == pc && !pending => f,
                _ => {
                    blocks.stats.chain_stops[if pending { 2 } else { 1 }] += 1;
                    break;
                }
            };
            st.budget = budget - total;
            if st.budget <= b.entry_need {
                blocks.stats.chain_stops[4] += 1;
                break;
            }
            st.blk = b;
            // SAFETY: `f` was compiled for `b`, and `st` is filled.
            unsafe { f(&raw mut st) };
            runs += 1;
            ops += st.ops as u64;
            total += st.acc;
            last_pc = st.last_pc;
            last_cycles = st.last_cycles;
            let (faulted, pc1) = unsafe { ((*busp).bus_fault() || (*me).pending_fault.is_some(), (*me).regs.r[15]) };
            let stop = if cx.exit || faulted {
                3
            } else if total >= budget {
                4
            } else if !(st.ops as usize == b.ops.len() && b.chain || side_exit(b, st.ops as usize)) {
                5
            } else {
                0
            };
            if stop != 0 {
                blocks.stats.chain_stops[stop] += 1;
                break;
            }
            if ctl.track && pc1 < last_pc {
                let now = ctl.clock + total - last_cycles;
                let no_op = match ctl.snap {
                    Some((pc, sp, lr)) => {
                        // SAFETY: plain reads, as above.
                        let (r, dirty) = unsafe { (&(*me).regs.r, (*busp).idle.dirty) };
                        !dirty && (pc1 != pc || r[13] != sp || r[14] != lr)
                    }
                    None => ctl.fails_maxed || crate::is_busy_head(ctl.busy, pc1, now),
                };
                if !no_op {
                    blocks.stats.chain_stops[6] += 1;
                    break;
                }
                // The visit `run_single` would make: it clears the flags and
                // drops nothing.
                unsafe {
                    (*busp).idle.dirty = false;
                    (*busp).idle.event = false;
                    (*busp).idle.exception = false;
                }
            }
        }
        if runs == 0 {
            return None;
        }
        self.current_instr_addr = last_pc;
        bus.set_active_pc(last_pc);
        let extra = self.retire_faults(bus, 0) as u64;
        total += extra;
        last_cycles += extra;
        self.cycles = self.cycles.wrapping_add(total);
        blocks.stats.runs += runs;
        blocks.stats.jit_runs += runs;
        blocks.stats.ops += ops;
        blocks.stats.io_exits += (cx.io != 0) as u64;
        blocks.stats.code_page_stores += cx.code_stores as u64;
        blocks.stats.chains += 1;
        for (a, b) in blocks.stats.jit_slow.iter_mut().zip(cx.slow_kind) {
            *a += b;
        }
        blocks.stats.jit_tracked += cx.tracked;
        blocks.stats.chained += runs;
        if !fast {
            blocks.stats.slow_ops += ops;
        }
        Some(Exit { cycles: total, last_pc, ops: ops as u32, last_cycles, fast, io: cx.io, code_stores: cx.code_stores })
    }

    /// Runs `blk`'s compiled code; the same contract as [`Self::exec`]. In
    /// check mode every access takes the slow path, which logs it.
    #[cfg(feature = "jit")]
    fn run_jit<const CHECK: bool>(
        &mut self,
        bus: &mut Bus,
        f: jit::JitFn,
        blk: &Block,
        budget: u64,
        log: &mut CheckLog,
        stats: &mut BlockStats,
    ) -> (Exit, Option<Upto>) {
        let fast = bus.jit_fast();
        let pages_ok = bus.jit_pages_ok() && !CHECK;
        let mut cx = Ctx {
            #[cfg(feature = "jit")]
            slow_kind: [0; 6],
            #[cfg(feature = "jit")]
            tracked: 0,
            fast, exit: false, other: 1 - self.core_id as usize, io: 0, code_stores: 0 };
        let mut upto = None;
        let core: *mut Self = self;
        let busp: *mut Bus = bus;
        // SAFETY: the pointers come from the exclusive borrows above and are
        // used only by `f` and the helpers it calls, before any of the
        // borrows is used again.
        let mut st = unsafe {
            jit::JitState {
                regs: &raw mut (*core).regs,
                sram: (*busp).memory.sram_ptr(),
                pages: if pages_ok { (*busp).jit_pages.as_ptr() } else { jit::ALL_SLOW.as_ptr() },
                bank: &raw mut (*busp).core0_bank_touched,
                budget,
                acc: 0,
                last_cycles: 0,
                last_pc: 0,
                ops: 0,
                core,
                bus: busp,
                cx: &raw mut cx,
                log,
                blk,
                upto: &raw mut upto,
                check: CHECK,
                timer_ok: 0,
                timer: [0; 2],
                track: 0,
            }
        };
        st.set_mode(bus, pages_ok);
        // SAFETY: `f` was compiled for a block with `blk`'s ops and takes a
        // filled `JitState`.
        unsafe { f(&raw mut st) };
        let ipc = st.last_pc;
        self.current_instr_addr = ipc;
        bus.set_active_pc(ipc);
        let faulted = bus.bus_fault() || self.pending_fault.is_some();
        if CHECK && upto.is_none() && !faulted {
            upto = Some(Upto { n: st.ops as usize, regs: self.regs, pc: self.regs.r[15], cycles: st.acc, io: log.io.len() });
        }
        let exit = Exit { cycles: st.acc, last_pc: ipc, ops: st.ops, last_cycles: st.last_cycles, fast, io: cx.io, code_stores: cx.code_stores };
        for (a, b) in stats.jit_slow.iter_mut().zip(cx.slow_kind) {
            *a += b;
        }
        stats.jit_tracked += cx.tracked;
        (exit, upto)
    }

    /// Runs `blk` from its entry. Returns the exit and, for check mode, the
    /// number of instructions and registers, cycles and log lengths to
    /// compare, taken before a final interpreter op or a fault.
    #[inline(always)]
    fn exec<const CHECK: bool>(&mut self, bus: &mut Bus, blk: &Block, budget: u64, log: &mut CheckLog) -> (Exit, Option<Upto>) {
        let base = blk.pc;
        let mut cx = Ctx {
            #[cfg(feature = "jit")]
            slow_kind: [0; 6],
            #[cfg(feature = "jit")]
            tracked: 0,
            fast: bus.jit_fast(), exit: false, other: 1 - self.core_id as usize, io: 0, code_stores: 0 };
        let mut acc: u64 = 0;
        let mut upto = None;
        let n = blk.ops.len();
        let mut i = 0usize;
        for &op in blk.ops.iter() {
            let ipc = base.wrapping_add(op.off as u32);
            // Masked so the register-file indexing needs no bounds checks.
            let d = (op.d & 15) as usize;
            let rn = (op.n & 15) as usize;
            let m = (op.m & 15) as usize;
            let imm = op.imm;
            let c: u32 = match op.k {
                K::LslI => {
                    let v = self.regs.r[m];
                    if imm == 0 {
                        self.regs.r[d] = v;
                        self.regs.set_nz(v);
                    } else {
                        let res = v << imm;
                        self.regs.r[d] = res;
                        self.regs.set_nz(res);
                        self.regs.set_flag_c((v >> (32 - imm)) & 1 != 0);
                    }
                    1
                }
                K::LsrI => {
                    let v = self.regs.r[m];
                    let (res, c) = if imm == 0 { (0, v >> 31 != 0) } else { (v >> imm, (v >> (imm - 1)) & 1 != 0) };
                    self.regs.r[d] = res;
                    self.regs.set_nz(res);
                    self.regs.set_flag_c(c);
                    1
                }
                K::AsrI => {
                    let v = self.regs.r[m] as i32;
                    let (res, c) = if imm == 0 {
                        ((v >> 31) as u32, v < 0)
                    } else {
                        ((v >> imm) as u32, (v >> (imm as i32 - 1)) & 1 != 0)
                    };
                    self.regs.r[d] = res;
                    self.regs.set_nz(res);
                    self.regs.set_flag_c(c);
                    1
                }
                K::AddR => {
                    let (res, c, v) = add_with_carry(self.regs.r[rn], self.regs.r[m], false);
                    self.regs.r[d] = res;
                    self.regs.set_nzcv(res >> 31 != 0, res == 0, c, v);
                    1
                }
                K::SubR => {
                    let (res, c, v) = add_with_carry(self.regs.r[rn], !self.regs.r[m], true);
                    self.regs.r[d] = res;
                    self.regs.set_nzcv(res >> 31 != 0, res == 0, c, v);
                    1
                }
                K::AddI => {
                    let (res, c, v) = add_with_carry(self.regs.r[rn], imm, false);
                    self.regs.r[d] = res;
                    self.regs.set_nzcv(res >> 31 != 0, res == 0, c, v);
                    1
                }
                K::SubI => {
                    let (res, c, v) = add_with_carry(self.regs.r[rn], !imm, true);
                    self.regs.r[d] = res;
                    self.regs.set_nzcv(res >> 31 != 0, res == 0, c, v);
                    1
                }
                K::MovI => {
                    self.regs.r[d] = imm;
                    self.regs.set_nz(imm);
                    1
                }
                K::CmpI => {
                    let (res, c, v) = add_with_carry(self.regs.r[rn], !imm, true);
                    self.regs.set_nzcv(res >> 31 != 0, res == 0, c, v);
                    1
                }
                K::And => {
                    let res = self.regs.r[d] & self.regs.r[m];
                    self.regs.r[d] = res;
                    self.regs.set_nz(res);
                    1
                }
                K::Eor => {
                    let res = self.regs.r[d] ^ self.regs.r[m];
                    self.regs.r[d] = res;
                    self.regs.set_nz(res);
                    1
                }
                K::Orr => {
                    let res = self.regs.r[d] | self.regs.r[m];
                    self.regs.r[d] = res;
                    self.regs.set_nz(res);
                    1
                }
                K::Bic => {
                    let res = self.regs.r[d] & !self.regs.r[m];
                    self.regs.r[d] = res;
                    self.regs.set_nz(res);
                    1
                }
                K::Mvn => {
                    let res = !self.regs.r[m];
                    self.regs.r[d] = res;
                    self.regs.set_nz(res);
                    1
                }
                K::Mul => {
                    let res = self.regs.r[d].wrapping_mul(self.regs.r[m]);
                    self.regs.r[d] = res;
                    self.regs.set_nz(res);
                    1
                }
                K::Tst => {
                    let res = self.regs.r[d] & self.regs.r[m];
                    self.regs.set_nz(res);
                    1
                }
                K::LslR => {
                    let a = self.regs.r[d];
                    let s = self.regs.r[m] & 0xFF;
                    let (res, c) = if s == 0 {
                        (a, self.regs.flag_c())
                    } else if s < 32 {
                        (a << s, (a >> (32 - s)) & 1 != 0)
                    } else if s == 32 {
                        (0, a & 1 != 0)
                    } else {
                        (0, false)
                    };
                    self.regs.r[d] = res;
                    self.regs.set_nz(res);
                    self.regs.set_flag_c(c);
                    1
                }
                K::LsrR => {
                    let a = self.regs.r[d];
                    let s = self.regs.r[m] & 0xFF;
                    let (res, c) = if s == 0 {
                        (a, self.regs.flag_c())
                    } else if s < 32 {
                        (a >> s, (a >> (s - 1)) & 1 != 0)
                    } else if s == 32 {
                        (0, a >> 31 != 0)
                    } else {
                        (0, false)
                    };
                    self.regs.r[d] = res;
                    self.regs.set_nz(res);
                    self.regs.set_flag_c(c);
                    1
                }
                K::AsrR => {
                    let a = self.regs.r[d];
                    let s = self.regs.r[m] & 0xFF;
                    let sa = a as i32;
                    let (res, c) = if s == 0 {
                        (a, self.regs.flag_c())
                    } else if s < 32 {
                        ((sa >> s) as u32, (sa >> (s as i32 - 1)) & 1 != 0)
                    } else {
                        ((sa >> 31) as u32, sa < 0)
                    };
                    self.regs.r[d] = res;
                    self.regs.set_nz(res);
                    self.regs.set_flag_c(c);
                    1
                }
                K::Ror => {
                    let a = self.regs.r[d];
                    let s = self.regs.r[m] & 0xFF;
                    let (res, c) = if s == 0 {
                        (a, self.regs.flag_c())
                    } else if s & 31 == 0 {
                        (a, a >> 31 != 0)
                    } else {
                        let x = a.rotate_right(s & 31);
                        (x, x >> 31 != 0)
                    };
                    self.regs.r[d] = res;
                    self.regs.set_nz(res);
                    self.regs.set_flag_c(c);
                    1
                }
                K::Adc => {
                    let cin = self.regs.flag_c();
                    let (res, c, v) = add_with_carry(self.regs.r[d], self.regs.r[m], cin);
                    self.regs.r[d] = res;
                    self.regs.set_nzcv(res >> 31 != 0, res == 0, c, v);
                    1
                }
                K::Sbc => {
                    let cin = self.regs.flag_c();
                    let (res, c, v) = add_with_carry(self.regs.r[d], !self.regs.r[m], cin);
                    self.regs.r[d] = res;
                    self.regs.set_nzcv(res >> 31 != 0, res == 0, c, v);
                    1
                }
                K::Neg => {
                    let (res, c, v) = add_with_carry(0, !self.regs.r[m], true);
                    self.regs.r[d] = res;
                    self.regs.set_nzcv(res >> 31 != 0, res == 0, c, v);
                    1
                }
                K::CmpR | K::CmpHi => {
                    let (res, c, v) = add_with_carry(self.regs.r[rn], !self.regs.r[m], true);
                    self.regs.set_nzcv(res >> 31 != 0, res == 0, c, v);
                    1
                }
                K::Cmn => {
                    let (res, c, v) = add_with_carry(self.regs.r[d], self.regs.r[m], false);
                    self.regs.set_nzcv(res >> 31 != 0, res == 0, c, v);
                    1
                }
                K::AddHi => {
                    self.regs.r[d] = self.regs.r[d].wrapping_add(self.regs.r[m]);
                    1
                }
                K::MovHi => {
                    self.regs.r[d] = self.regs.r[m];
                    1
                }
                K::MovK => {
                    self.regs.r[d] = imm;
                    1
                }
                K::AddSpK => {
                    self.regs.r[d] = self.regs.r[13].wrapping_add(imm);
                    1
                }
                K::AddSp => {
                    self.regs.r[13] = self.regs.r[13].wrapping_add(imm);
                    1
                }
                K::SubSp => {
                    self.regs.r[13] = self.regs.r[13].wrapping_sub(imm);
                    1
                }
                K::Sxth => {
                    self.regs.r[d] = self.regs.r[m] as i16 as i32 as u32;
                    1
                }
                K::Sxtb => {
                    self.regs.r[d] = self.regs.r[m] as i8 as i32 as u32;
                    1
                }
                K::Uxth => {
                    self.regs.r[d] = self.regs.r[m] & 0xFFFF;
                    1
                }
                K::Uxtb => {
                    self.regs.r[d] = self.regs.r[m] & 0xFF;
                    1
                }
                K::Rev => {
                    self.regs.r[d] = self.regs.r[m].swap_bytes();
                    1
                }
                K::Rev16 => {
                    let v = self.regs.r[m];
                    self.regs.r[d] = ((v >> 8) & 0x00FF_00FF) | ((v << 8) & 0xFF00_FF00);
                    1
                }
                K::Revsh => {
                    self.regs.r[d] = (self.regs.r[m] as u16).swap_bytes() as i16 as i32 as u32;
                    1
                }
                K::Nop => 1,
                K::LdrK => {
                    let v = Self::ld32::<CHECK>(bus, &mut cx, log, imm, ipc);
                    self.regs.r[d] = v;
                    2
                }
                K::LdrI | K::LdrR => {
                    let addr = self.regs.r[rn].wrapping_add(if op.k == K::LdrI { imm } else { self.regs.r[m] });
                    if !is_aligned(addr, 4) {
                        self.pending_fault = Some(Fault::Unaligned);
                        cx.exit = true;
                        1
                    } else {
                        let v = Self::ld32::<CHECK>(bus, &mut cx, log, addr, ipc);
                        self.regs.r[d] = v;
                        2
                    }
                }
                K::StrI | K::StrR => {
                    let addr = self.regs.r[rn].wrapping_add(if op.k == K::StrI { imm } else { self.regs.r[m] });
                    if !is_aligned(addr, 4) {
                        self.pending_fault = Some(Fault::Unaligned);
                        cx.exit = true;
                        1
                    } else {
                        let v = self.regs.r[d];
                        Self::st32::<CHECK>(bus, &mut cx, log, addr, v, ipc);
                        2
                    }
                }
                K::LdrhI | K::LdrhR | K::LdrshR => {
                    let addr = self.regs.r[rn].wrapping_add(if op.k == K::LdrhI { imm } else { self.regs.r[m] });
                    if !is_aligned(addr, 2) {
                        self.pending_fault = Some(Fault::Unaligned);
                        cx.exit = true;
                        1
                    } else {
                        let v = Self::ld16::<CHECK>(bus, &mut cx, log, addr, ipc);
                        self.regs.r[d] = if op.k == K::LdrshR { v as u16 as i16 as i32 as u32 } else { v };
                        2
                    }
                }
                K::StrhI | K::StrhR => {
                    let addr = self.regs.r[rn].wrapping_add(if op.k == K::StrhI { imm } else { self.regs.r[m] });
                    if !is_aligned(addr, 2) {
                        self.pending_fault = Some(Fault::Unaligned);
                        cx.exit = true;
                        1
                    } else {
                        let v = self.regs.r[d];
                        Self::st16::<CHECK>(bus, &mut cx, log, addr, v, ipc);
                        2
                    }
                }
                K::LdrbI | K::LdrbR | K::LdrsbR => {
                    let addr = self.regs.r[rn].wrapping_add(if op.k == K::LdrbI { imm } else { self.regs.r[m] });
                    let v = Self::ld8::<CHECK>(bus, &mut cx, log, addr, ipc);
                    self.regs.r[d] = if op.k == K::LdrsbR { v as u8 as i8 as i32 as u32 } else { v };
                    2
                }
                K::StrbI | K::StrbR => {
                    let addr = self.regs.r[rn].wrapping_add(if op.k == K::StrbI { imm } else { self.regs.r[m] });
                    let v = self.regs.r[d];
                    Self::st8::<CHECK>(bus, &mut cx, log, addr, v, ipc);
                    2
                }
                K::Push => {
                    let count = imm.count_ones();
                    let base = self.regs.r[13].wrapping_sub(count * 4);
                    if !is_aligned(base, 4) {
                        self.pending_fault = Some(Fault::Unaligned);
                        cx.exit = true;
                        1
                    } else {
                        self.regs.r[13] = base;
                        let mut a = base;
                        let mut list = imm;
                        while list != 0 {
                            let reg = list.trailing_zeros() as usize;
                            list &= list - 1;
                            let v = self.regs.r[reg];
                            Self::st32::<CHECK>(bus, &mut cx, log, a, v, ipc);
                            a = a.wrapping_add(4);
                        }
                        1 + count
                    }
                }
                K::Pop => {
                    let count = imm.count_ones();
                    let sp = self.regs.r[13];
                    if !is_aligned(sp, 4) {
                        self.pending_fault = Some(Fault::Unaligned);
                        cx.exit = true;
                        1
                    } else {
                        let mut a = sp;
                        let mut list = imm;
                        while list != 0 {
                            let reg = list.trailing_zeros() as usize;
                            list &= list - 1;
                            let v = Self::ld32::<CHECK>(bus, &mut cx, log, a, ipc);
                            self.regs.r[reg] = v;
                            a = a.wrapping_add(4);
                        }
                        self.regs.r[13] = a;
                        1 + count
                    }
                }
                K::Stm => {
                    let count = imm.count_ones();
                    let mut a = self.regs.r[rn];
                    if !is_aligned(a, 4) {
                        self.pending_fault = Some(Fault::Unaligned);
                        cx.exit = true;
                        1
                    } else {
                        let mut list = imm;
                        while list != 0 {
                            let reg = list.trailing_zeros() as usize;
                            list &= list - 1;
                            let v = self.regs.r[reg];
                            Self::st32::<CHECK>(bus, &mut cx, log, a, v, ipc);
                            a = a.wrapping_add(4);
                        }
                        self.regs.r[rn] = a;
                        1 + count
                    }
                }
                K::Ldm => {
                    let count = imm.count_ones();
                    let mut a = self.regs.r[rn];
                    if !is_aligned(a, 4) {
                        self.pending_fault = Some(Fault::Unaligned);
                        cx.exit = true;
                        1
                    } else {
                        let mut list = imm;
                        while list != 0 {
                            let reg = list.trailing_zeros() as usize;
                            list &= list - 1;
                            let v = Self::ld32::<CHECK>(bus, &mut cx, log, a, ipc);
                            self.regs.r[reg] = v;
                            a = a.wrapping_add(4);
                        }
                        if imm & (1 << rn) == 0 {
                            self.regs.r[rn] = a;
                        }
                        1 + count
                    }
                }
                K::BCond => {
                    if self.regs.condition_passed(m as u8) {
                        self.regs.r[15] = imm;
                        3
                    } else {
                        self.regs.r[15] = ipc.wrapping_add(2);
                        1
                    }
                }
                K::B => {
                    self.regs.r[15] = imm;
                    3
                }
                K::Bl => {
                    self.regs.r[14] = ipc.wrapping_add(4) | 1;
                    self.regs.r[15] = imm;
                    4
                }
                K::Interp | K::InterpEnd => {
                    if CHECK && op.k == K::InterpEnd {
                        upto = Some(Upto { n: i, regs: self.regs, pc: ipc, cycles: acc, io: log.io.len() });
                    }
                    let len = op.len();
                    self.current_instr_addr = ipc;
                    self.regs.set_pc(ipc.wrapping_add(len));
                    bus.set_active_pc(ipc);
                    let c = if len == 4 {
                        self.execute_thumb32(op.hw0, imm as u16, bus)
                    } else {
                        self.execute_thumb16(op.hw0, bus)
                    };
                    cx.exit |= self.pending_fault.is_some();
                    c
                }
            };
            let before = acc;
            acc += c as u64;
            i += 1;
            let ends = op.k.ends_block();
            // A branch before the last op leaves the block when it does not
            // go to the next op; faults set `cx.exit`.
            let left = ends && i < n && self.regs.r[15] != base.wrapping_add(blk.ops[i].off as u32);
            if i == n || cx.exit || acc >= budget || left {
                if !ends {
                    self.regs.r[15] = ipc.wrapping_add(op.len());
                }
                self.current_instr_addr = ipc;
                bus.set_active_pc(ipc);
                // A prefix taken before a final interpreter op stays
                // comparable even when that op faults.
                let faulted = bus.bus_fault() || self.pending_fault.is_some();
                if CHECK && upto.is_none() && !faulted {
                    upto = Some(Upto { n: i, regs: self.regs, pc: self.regs.r[15], cycles: acc, io: log.io.len() });
                }
                // Fault entry, if any, runs after check mode has compared.
                return (Exit { cycles: acc, last_pc: ipc, ops: i as u32, last_cycles: acc - before, fast: cx.fast, io: cx.io, code_stores: cx.code_stores }, upto);
            }
        }
        unreachable!("a block always exits at its last op")
    }

    /// Re-runs the first `upto.n` instructions of `blk` on the interpreter
    /// against a sandbox holding the memory as it was at block entry, and
    /// compares the result with what the block did.
    fn check_block(&mut self, bus: &mut Bus, blocks: &mut Blocks, blk: &Block, pre: Registers, upto: Option<Upto>) {
        let Some(u) = upto else {
            blocks.stats.unchecked += 1;
            return;
        };
        blocks.stats.checked += 1;
        if u.n == 0 {
            return;
        }
        let core = self.core_id as usize;
        let scratch = blocks.scratch.get_or_insert_with(|| Box::new(CortexM0Plus::with_id(0)));
        scratch.core_id = self.core_id;
        scratch.regs = pre;
        scratch.pending_fault = None;
        scratch.halted = false;
        let mut undo: HashMap<u32, u8> = HashMap::new();
        for &(off, size, old) in blocks.log.sram_old.iter().rev() {
            for b in 0..size as u32 {
                undo.insert(off + b, (old >> (8 * b)) as u8);
            }
        }
        let mut sb = Sandbox {
            bus,
            undo: &undo,
            overlay: HashMap::new(),
            io: &blocks.log.io[..u.io],
            io_pos: 0,
            err: None,
            ppb: bus.ppb[core].clone(),
            nvic: bus.nvics[core],
            core,
        };
        let mut cycles = 0u64;
        for _ in 0..u.n {
            let pc = scratch.regs.pc();
            let hw0 = sb.code16(pc);
            let wide = is_wide(hw0);
            scratch.current_instr_addr = pc;
            if wide {
                let hw1 = sb.code16(pc + 2);
                scratch.regs.set_pc(pc + 4);
                cycles += scratch.execute_thumb32(hw0, hw1, &mut sb) as u64;
            } else {
                scratch.regs.set_pc(pc + 2);
                cycles += scratch.execute_thumb16(hw0, &mut sb) as u64;
            }
        }
        let mut diffs = Vec::new();
        if let Some(e) = sb.err.take() {
            diffs.push(e);
        }
        if sb.io_pos != sb.io.len() {
            diffs.push(format!("block made {} MMIO accesses, interpreter {}", sb.io.len(), sb.io_pos));
        }
        let mut regs = u.regs;
        regs.r[15] = u.pc;
        if scratch.regs != regs {
            for k in 0..16 {
                if scratch.regs.r[k] != regs.r[k] {
                    diffs.push(format!("r{k}: block {:#010x} interp {:#010x}", regs.r[k], scratch.regs.r[k]));
                }
            }
            if scratch.regs.xpsr != regs.xpsr {
                diffs.push(format!("xpsr: block {:#010x} interp {:#010x}", regs.xpsr, scratch.regs.xpsr));
            }
            if (scratch.regs.primask, scratch.regs.control, scratch.regs.msp, scratch.regs.psp)
                != (regs.primask, regs.control, regs.msp, regs.psp)
            {
                diffs.push("primask, control or banked SP differ".into());
            }
        }
        if scratch.pending_fault.is_some() {
            diffs.push(format!("interpreter faulted: {:?}", scratch.pending_fault));
        }
        if cycles != u.cycles {
            diffs.push(format!("cycles: block {} interp {}", u.cycles, cycles));
        }
        // Stores: the interpreter's overlay must equal memory now, and every
        // byte the block stored must be in the overlay.
        for (&off, &v) in &sb.overlay {
            let now = sb.bus.memory.sram_read8(off);
            if now != v {
                diffs.push(format!("sram {:#010x}: block {:#04x} interp {:#04x}", 0x2000_0000 + off, now, v));
            }
        }
        for &off in undo.keys() {
            if !sb.overlay.contains_key(&off) {
                diffs.push(format!("sram {:#010x}: stored by block only", 0x2000_0000 + off));
            }
        }
        if !diffs.is_empty() {
            blocks.stats.mismatches += 1;
            if blocks.reports.len() < 8 {
                let mut s = format!("block check mismatch: core {core} block {:#010x}, {} of {} ops run\n", blk.pc, u.n, blk.ops.len());
                s += &format!("  entry registers {:08x?} xpsr {:08x}\n", pre.r, pre.xpsr);
                for d in diffs.iter().take(24) {
                    s += &format!("  {d}\n");
                }
                for op in blk.ops.iter() {
                    s += &format!("  {:#010x} {:04x} {:?}\n", blk.pc + op.off as u32, op.hw0, op);
                }
                blocks.reports.push(s);
            }
            // A lockstep check fails loudly in any harness.
            panic!("{}", blocks.reports.last().map(String::as_str).unwrap_or("block check mismatch"));
        }
    }

    /// Delivers a fault the last instruction raised, as `step` does after
    /// `decode_execute`; returns `cycles` plus the fault entry.
    pub(crate) fn retire_faults<B: CoreBus>(&mut self, bus: &mut B, mut cycles: u32) -> u32 {
        if bus.bus_fault() {
            tracing::info!(
                pc = format_args!("{:#010x}", self.current_instr_addr),
                addr = format_args!("{:#010x}", bus.bus_fault_addr()),
                "HardFault escalation from bus fault"
            );
            bus.clear_bus_fault();
            self.pending_fault = Some(Fault::HardFault);
        }
        if let Some(fault) = self.pending_fault.take() {
            cycles = cycles.wrapping_add(self.deliver_fault(fault, bus));
        }
        cycles
    }
}

/// What check mode compares: the state after the first `n` instructions.
struct Upto {
    n: usize,
    regs: Registers,
    pc: u32,
    cycles: u64,
    io: usize,
}

/// The bus as the interpreter sees it when re-running a block: SRAM as it
/// was at block entry plus its own stores, MMIO reads replayed from the
/// block's log, MMIO stores compared with it.
struct Sandbox<'a> {
    bus: &'a Bus,
    undo: &'a HashMap<u32, u8>,
    overlay: HashMap<u32, u8>,
    io: &'a [(bool, u32, u8, u32)],
    io_pos: usize,
    err: Option<String>,
    ppb: Ppb,
    nvic: Nvic,
    core: usize,
}

impl Sandbox<'_> {
    fn sram_byte(&self, off: u32) -> u8 {
        if let Some(&v) = self.overlay.get(&off) {
            return v;
        }
        if let Some(&v) = self.undo.get(&off) {
            return v;
        }
        self.bus.memory.sram_read8(off)
    }

    fn code16(&self, pc: u32) -> u16 {
        if pc >> 24 == 0x20 {
            let off = pc & 0x00FF_FFFF;
            self.sram_byte(off) as u16 | (self.sram_byte(off + 1) as u16) << 8
        } else {
            self.bus.peek_code16(pc).unwrap_or(0)
        }
    }

    fn sram(addr: u32, size: u32) -> Option<u32> {
        let off = addr & 0x00FF_FFFF;
        (addr >> 24 == 0x20 && (off + size) as usize <= SRAM_SIZE).then_some(off)
    }

    fn read(&mut self, addr: u32, size: u32) -> u32 {
        if let Some(off) = Self::sram(addr, size) {
            return (0..size).fold(0, |v, b| v | (self.sram_byte(off + b) as u32) << (8 * b));
        }
        match self.io.get(self.io_pos) {
            Some(&(false, a, s, v)) if a == addr && s as u32 == size => {
                self.io_pos += 1;
                v
            }
            other => {
                self.err.get_or_insert_with(|| format!("interpreter read {addr:#010x} size {size}, block log has {other:?}"));
                0
            }
        }
    }

    fn write(&mut self, addr: u32, size: u32, val: u32) {
        if let Some(off) = Self::sram(addr, size) {
            for b in 0..size {
                self.overlay.insert(off + b, (val >> (8 * b)) as u8);
            }
            return;
        }
        match self.io.get(self.io_pos) {
            Some(&(true, a, s, v)) if a == addr && s as u32 == size && v == val => self.io_pos += 1,
            other => {
                self.err
                    .get_or_insert_with(|| format!("interpreter wrote {val:#x} to {addr:#010x} size {size}, block log has {other:?}"));
            }
        }
    }
}

impl CoreBus for Sandbox<'_> {
    fn read8(&mut self, addr: u32) -> u8 {
        self.read(addr, 1) as u8
    }
    fn read16(&mut self, addr: u32) -> u16 {
        self.read(addr, 2) as u16
    }
    fn read32(&mut self, addr: u32) -> u32 {
        self.read(addr, 4)
    }
    fn write8(&mut self, addr: u32, val: u8) {
        self.write(addr, 1, val as u32)
    }
    fn write16(&mut self, addr: u32, val: u16) {
        self.write(addr, 2, val as u32)
    }
    fn write32(&mut self, addr: u32, val: u32) {
        self.write(addr, 4, val)
    }
    fn set_active_pc(&mut self, _pc: u32) {}
    fn bus_fault(&self) -> bool {
        false
    }
    fn bus_fault_addr(&self) -> u32 {
        0
    }
    fn clear_bus_fault(&mut self) {}
    fn ppb(&self, _core: usize) -> &Ppb {
        &self.ppb
    }
    fn ppb_mut(&mut self, _core: usize) -> &mut Ppb {
        &mut self.ppb
    }
    fn nvic(&self, _core: usize) -> &Nvic {
        &self.nvic
    }
    fn nvic_mut(&mut self, _core: usize) -> &mut Nvic {
        &mut self.nvic
    }
    fn active_core(&self) -> usize {
        self.core
    }
    fn event_flag(&self, _core: usize) -> bool {
        false
    }
    fn consume_event_flag(&mut self, _core: usize) -> bool {
        false
    }
    fn set_wfe_waiting(&mut self, _core: usize, _val: bool) {}
    fn signal_sev(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every 16-bit encoding lowers to an op whose length matches the
    /// decoder's, and the branch kinds end their block.
    #[test]
    fn lower_covers_every_thumb16_encoding() {
        for hw0 in 0..=0xFFFFu16 {
            if is_wide(hw0) {
                continue;
            }
            let op = lower(hw0, 0, 0x2000_0100, 0);
            assert_eq!(op.len(), 2, "{hw0:04x}");
            if hw0 >> 11 == 0b11100 {
                assert_eq!(op.k, K::B);
            }
        }
    }

    #[test]
    fn bl_target_matches_interpreter() {
        // BL from 0x2000_0100 to 0x2000_0000: offset -0x104.
        let off: i32 = -0x104;
        let imm = (off as u32) >> 1;
        let s = (imm >> 23) & 1;
        let i1 = (imm >> 22) & 1;
        let i2 = (imm >> 21) & 1;
        let j1 = (!(i1 ^ s)) & 1;
        let j2 = (!(i2 ^ s)) & 1;
        let hw0 = (0xF000 | (s << 10) | ((imm >> 11) & 0x3FF)) as u16;
        let hw1 = (0xD000 | (j1 << 13) | (j2 << 11) | (imm & 0x7FF)) as u16;
        let op = lower(hw0, hw1, 0x2000_0100, 0);
        assert_eq!(op.k, K::Bl);
        assert_eq!(op.imm, 0x2000_0000);
        let mut c = CortexM0Plus::new();
        c.regs.set_pc(0x2000_0100);
        c.execute_one_wide(hw0, hw1);
        assert_eq!(c.regs.pc(), 0x2000_0000);
    }
}
