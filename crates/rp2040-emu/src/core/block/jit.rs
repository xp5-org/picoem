//! Cranelift backend for the block engine, Stage 2 of
//! `wrk_docs/2026.11.25 - HLD - RP2040 M0+ JIT V1.md`.
//!
//! A hot block becomes one host function that takes a [`JitState`]. It does
//! what [`CortexM0Plus::exec`] does for the same block, instruction for
//! instruction and cycle for cycle:
//!
//! - Guest registers and the N, Z, C and V flags live in SSA variables,
//!   loaded at entry and written back at the exit and around interpreter
//!   calls. Flags that nothing reads are never computed.
//! - An SRAM access whose page byte in [`Bus::jit_pages`] is clear reads or
//!   writes host memory directly; every other access calls [`h_mem`], which
//!   is the micro-op loop's slow path. With the fast path off the state
//!   points at a table of slow pages, so check mode logs every access.
//! - While idle tracking is live the fast path stays on: `track` in the
//!   state tells compiled code to call [`h_track`] before each fast SRAM
//!   access and inline TIMER read, which runs the idle detector's hooks the
//!   slow path runs in front of the bus.
//! - After each instruction the block exits when the cycle budget is
//!   reached or the slow path asked for an exit, with PC set as the
//!   interpreter leaves it.

use core::mem::offset_of;

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::{I8, I16, I32, I64};
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlagsData, SigRef, Signature, Type, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Module, default_libcall_names};

use super::{Block, CheckLog, Ctx, K, Op, Upto};
use crate::bus::Bus;
use crate::core::bus_trait::CoreBus;
use crate::core::{CortexM0Plus, Fault, Registers};
use crate::memory::SRAM_SIZE;

/// A compiled block.
pub(crate) type JitFn = unsafe extern "C" fn(*mut JitState);

/// Compiled blocks kept before the code memory is dropped and rebuilt.
pub(crate) const JIT_MAX_BLOCKS: usize = 16384;

/// Every page slow: the table used when the fast path is off.
pub(crate) static ALL_SLOW: [u8; crate::bus::JIT_PAGES] = [0xFF; crate::bus::JIT_PAGES];

/// What a compiled block reads and writes besides guest memory.
#[repr(C)]
pub(crate) struct JitState {
    pub regs: *mut Registers,
    pub sram: *mut u8,
    pub pages: *const u8,
    pub bank: *mut u8,
    pub budget: u64,
    pub acc: u64,
    pub last_cycles: u64,
    pub last_pc: u32,
    pub ops: u32,
    pub core: *mut CortexM0Plus,
    pub bus: *mut Bus,
    pub cx: *mut Ctx,
    pub log: *mut CheckLog,
    pub blk: *const Block,
    pub upto: *mut Option<Upto>,
    pub check: bool,
    /// TIMERAWL and TIMERAWH may be read from `timer` below: the fast path
    /// is on and TIMER is out of reset. The bus reads them from
    /// `master_cycle`, which holds still through a `run_single` call.
    pub timer_ok: u32,
    pub timer: [u32; 2],
    /// Idle-tracking hooks compiled code calls on its fast path:
    /// [`TRACK_READS`] and [`TRACK_STORES`].
    pub track: u32,
}

/// `JitState::track`: fast loads and inline TIMER reads call [`h_track`],
/// as watch-mode tracking classifies every read.
pub(crate) const TRACK_READS: u32 = 1;
/// `JitState::track`: fast stores call [`h_track`].
pub(crate) const TRACK_STORES: u32 = 2;

impl JitState {
    /// Fills the TIMER and tracking fields from `bus` for a run with the
    /// fast path `fast`.
    pub fn set_mode(&mut self, bus: &Bus, fast: bool) {
        let (ok, t) = bus.jit_timer();
        self.timer_ok = (fast && ok) as u32;
        self.timer = [t as u32, (t >> 32) as u32];
        // Outside watch mode every SRAM and TIMER read is clean, so only
        // stores need the hook.
        self.track = if fast && bus.idle.on { TRACK_STORES | if bus.idle.watch { TRACK_READS } else { 0 } } else { 0 };
    }
}

// ---------------------------------------------------------------------------
// Helpers called from compiled code
// ---------------------------------------------------------------------------

/// Slow-path access. `kind` is the size in bytes, plus 8 for a store.
/// Returns the loaded value, zero-extended, and bit 32 set when the block
/// must end after this instruction.
unsafe extern "C" fn h_mem(st: *mut JitState, addr: u32, val: u32, kind: u32, ipc: u32) -> u64 {
    // SAFETY: `run_jit` fills every pointer from live exclusive borrows
    // that outlast the call, and no Rust reference to them is held while
    // compiled code runs.
    let st = unsafe { &mut *st };
    let (bus, cx, log) = unsafe { (&mut *st.bus, &mut *st.cx, &mut *st.log) };
    cx.exit = false;
    let write = kind & 8 != 0;
    let why = if st.pages == ALL_SLOW.as_ptr() {
        0
    } else if addr.wrapping_sub(0x2000_0000) < SRAM_SIZE as u32 {
        1
    } else {
        2
    };
    cx.slow_kind[2 * why + write as usize] += 1;
    let v = if st.check { mem::<true>(bus, cx, log, addr, val, kind, ipc) } else { mem::<false>(bus, cx, log, addr, val, kind, ipc) };
    v as u64 | (cx.exit as u64) << 32
}

#[inline(always)]
fn mem<const C: bool>(bus: &mut Bus, cx: &mut Ctx, log: &mut CheckLog, addr: u32, val: u32, kind: u32, ipc: u32) -> u32 {
    type S = CortexM0Plus;
    match kind {
        1 => S::slow::<C>(bus, cx, log, addr, 1, None, ipc, |b, a, _| CoreBus::read8(b, a) as u32),
        2 => S::slow::<C>(bus, cx, log, addr, 2, None, ipc, |b, a, _| CoreBus::read16(b, a) as u32),
        4 => S::slow::<C>(bus, cx, log, addr, 4, None, ipc, |b, a, _| CoreBus::read32(b, a)),
        9 => S::slow::<C>(bus, cx, log, addr, 1, Some(val), ipc, |b, a, v| {
            CoreBus::write8(b, a, v as u8);
            0
        }),
        10 => S::slow::<C>(bus, cx, log, addr, 2, Some(val), ipc, |b, a, v| {
            CoreBus::write16(b, a, v as u16);
            0
        }),
        _ => S::slow::<C>(bus, cx, log, addr, 4, Some(val), ipc, |b, a, v| {
            CoreBus::write32(b, a, v);
            0
        }),
    }
}

/// Idle tracking for a fast-path access the bus does not see: the
/// detector's part of the hook [`CoreBus`] runs in front of the same
/// access; the park check cannot hit on the fast path. `kind` is as for
/// [`h_mem`]. A store's hook runs before the store, so it logs the value
/// overwritten.
unsafe extern "C" fn h_track(st: *mut JitState, addr: u32, kind: u32) {
    // SAFETY: as in `h_mem`.
    let st = unsafe { &mut *st };
    let (bus, cx) = unsafe { (&mut *st.bus, &mut *st.cx) };
    cx.tracked += 1;
    if kind & 8 != 0 {
        bus.idle_classify_write(addr, kind & 7);
    } else {
        bus.idle_classify_read(addr);
    }
}

/// An unaligned access: the fault the micro-op loop records.
unsafe extern "C" fn h_fault(st: *mut JitState) {
    // SAFETY: as in `h_mem`.
    unsafe { (*(*st).core).pending_fault = Some(Fault::Unaligned) };
}

/// Runs op `idx` on the interpreter's executor, registers and flags
/// already written back. Returns its cycles and bit 32 on a fault.
unsafe extern "C" fn h_interp(st: *mut JitState, idx: u32, acc: u64) -> u64 {
    // SAFETY: as in `h_mem`.
    let st = unsafe { &mut *st };
    let (core, bus, blk) = unsafe { (&mut *st.core, &mut *st.bus, &*st.blk) };
    let op = blk.ops[idx as usize];
    let ipc = blk.pc.wrapping_add(op.off as u32);
    if st.check && op.k == K::InterpEnd {
        let io = unsafe { (*st.log).io.len() };
        unsafe { *st.upto = Some(Upto { n: idx as usize, regs: core.regs, pc: ipc, cycles: acc, io }) };
    }
    let len = op.len();
    core.current_instr_addr = ipc;
    core.regs.set_pc(ipc.wrapping_add(len));
    bus.set_active_pc(ipc);
    let c = if len == 4 { core.execute_thumb32(op.hw0, op.imm as u16, bus) } else { core.execute_thumb16(op.hw0, bus) };
    c as u64 | (core.pending_fault.is_some() as u64) << 32
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

/// The code memory and compiler state for one emulator.
pub(crate) struct Jit {
    module: JITModule,
    ctx: Context,
    fctx: FunctionBuilderContext,
    sig_mem: Signature,
    sig_interp: Signature,
    sig_fault: Signature,
    sig_track: Signature,
    /// Blocks compiled into this module.
    pub compiled: usize,
    /// `/tmp/perf-PID.map` when `PICOEM_JIT_PERFMAP` is set, so `perf
    /// report` names compiled blocks.
    perf_map: Option<std::fs::File>,
}

// SAFETY: the module owns its code memory and holds no thread-bound state;
// the emulator using it is moved between threads, never shared.
unsafe impl Send for Jit {}

impl Jit {
    pub fn new() -> Option<Self> {
        let mut fb = settings::builder();
        fb.set("use_colocated_libcalls", "false").ok()?;
        fb.set("is_pic", "false").ok()?;
        // Blocks average five instructions: the optimiser buys nothing at
        // run time and doubles the compile time. `PICOEM_JIT_OPT=speed`
        // turns it on, `fast` picks the single-pass register allocator.
        let opt = std::env::var("PICOEM_JIT_OPT").unwrap_or_default();
        fb.set("opt_level", if opt.contains("speed") { "speed" } else { "none" }).ok()?;
        if opt.contains("fast") {
            fb.set("regalloc_algorithm", "single_pass").ok()?;
        }
        // Translations are checked against the interpreter instead; the IR
        // verifier would double the compile time.
        fb.set("enable_verifier", "false").ok()?;
        let isa = cranelift_native::builder().ok()?.finish(settings::Flags::new(fb)).ok()?;
        let module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));
        let ptr = module.target_config().pointer_type();
        let mut sig_mem = module.make_signature();
        for t in [ptr, I32, I32, I32, I32] {
            sig_mem.params.push(AbiParam::new(t));
        }
        sig_mem.returns.push(AbiParam::new(I64));
        let mut sig_interp = module.make_signature();
        for t in [ptr, I32, I64] {
            sig_interp.params.push(AbiParam::new(t));
        }
        sig_interp.returns.push(AbiParam::new(I64));
        let mut sig_fault = module.make_signature();
        sig_fault.params.push(AbiParam::new(ptr));
        let mut sig_track = module.make_signature();
        for t in [ptr, I32, I32] {
            sig_track.params.push(AbiParam::new(t));
        }
        let ctx = module.make_context();
        let perf_map = std::env::var_os("PICOEM_JIT_PERFMAP").and_then(|_| {
            std::fs::OpenOptions::new().create(true).append(true).open(format!("/tmp/perf-{}.map", std::process::id())).ok()
        });
        Some(Self { module, ctx, fctx: FunctionBuilderContext::new(), sig_mem, sig_interp, sig_fault, sig_track, compiled: 0, perf_map })
    }

    /// Compiles `blk`; `core0` adds core 0's bank-touch record to fast
    /// SRAM accesses, as [`Bus::jit_sram`] does.
    pub fn compile(&mut self, blk: &Block, core0: bool) -> Option<JitFn> {
        let tc = self.module.target_config();
        let ptr = tc.pointer_type();
        self.module.clear_context(&mut self.ctx);
        self.ctx.func.signature.params.push(AbiParam::new(ptr));
        let id = self.module.declare_anonymous_function(&self.ctx.func.signature).ok()?;
        {
            let mut b = FunctionBuilder::new(&mut self.ctx.func, &mut self.fctx);
            let sigs = [b.import_signature(self.sig_mem.clone()), b.import_signature(self.sig_interp.clone()), b.import_signature(self.sig_fault.clone()), b.import_signature(self.sig_track.clone())];
            Tr::emit(&mut b, blk, core0, ptr, sigs);
            b.seal_all_blocks();
            b.finalize(tc);
        }
        self.module.define_function(id, &mut self.ctx).ok()?;
        let size = self.ctx.compiled_code().map_or(0, |c| c.code_info().total_size);
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().ok()?;
        self.compiled += 1;
        let p = self.module.get_finalized_function(id);
        if let Some(f) = self.perf_map.as_mut() {
            use std::io::Write;
            let _ = writeln!(f, "{:x} {:x} jit_block_{:08x}", p as usize, size, blk.pc);
        }
        // SAFETY: `p` is the entry of a finalized function with the
        // signature `JitFn` describes.
        Some(unsafe { std::mem::transmute::<*const u8, JitFn>(p) })
    }
}

const fn st_off(o: usize) -> i32 {
    o as i32
}

/// Registers an op reads and writes, as bit masks over r0 to r14, and
/// whether it reads or writes the flags.
fn uses(op: &Op) -> (u16, u16, bool) {
    let (d, n, m) = (1u16 << (op.d & 15), 1u16 << (op.n & 15), 1u16 << (op.m & 15));
    let list = (op.imm & 0x7FFF) as u16;
    let (r, w, f) = match op.k {
        K::LslI | K::LsrI | K::AsrI => (m, d, true),
        K::AddR | K::SubR => (n | m, d, true),
        K::AddI | K::SubI => (n, d, true),
        K::MovI => (0, d, true),
        K::CmpI => (n, 0, true),
        K::And | K::Eor | K::Orr | K::Bic | K::Mul | K::Adc | K::Sbc | K::LslR | K::LsrR | K::AsrR | K::Ror => (d | m, d, true),
        K::Mvn | K::Neg => (m, d, true),
        K::Tst | K::Cmn => (d | m, 0, true),
        K::CmpR | K::CmpHi => (n | m, 0, true),
        K::AddHi => (d | m, d, false),
        K::MovHi | K::Sxth | K::Sxtb | K::Uxth | K::Uxtb | K::Rev | K::Rev16 | K::Revsh => (m, d, false),
        K::MovK | K::LdrK => (0, d, false),
        K::AddSpK => (1 << 13, d, false),
        K::AddSp | K::SubSp => (1 << 13, 1 << 13, false),
        K::StrR | K::StrhR | K::StrbR => (d | n | m, 0, false),
        K::LdrR | K::LdrhR | K::LdrbR | K::LdrsbR | K::LdrshR => (n | m, d, false),
        K::StrI | K::StrbI | K::StrhI => (d | n, 0, false),
        K::LdrI | K::LdrbI | K::LdrhI => (n, d, false),
        K::Push => (list | 1 << 13, 1 << 13, false),
        K::Pop => (1 << 13, list | 1 << 13, false),
        K::Stm => (list | n, n, false),
        K::Ldm => (n, list | n, false),
        K::BCond => (0, 0, true),
        K::Bl => (0, 1 << 14, false),
        K::B | K::Nop | K::Interp | K::InterpEnd => (0, 0, false),
    };
    (r & 0x7FFF, w & 0x7FFF, f)
}

/// One block's translation.
struct Tr<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    ptr: Type,
    st: Value,
    regs: Value,
    sram: Value,
    pages: Value,
    bank: Value,
    core0: bool,
    sigs: [SigRef; 4],
    r: [Variable; 15],
    /// N, Z, C, V as 0 or 1.
    f: [Variable; 4],
    acc: Variable,
    ex: Variable,
    used: u16,
    written: u16,
    flags: bool,
}

const MF: MemFlagsData = MemFlagsData::trusted();
const RO: i32 = offset_of!(Registers, r) as i32;
const XO: i32 = offset_of!(Registers, xpsr) as i32;

impl Tr<'_, '_> {
    fn emit(b: &mut FunctionBuilder<'_>, blk: &Block, core0: bool, ptr: Type, sigs: [SigRef; 4]) {
        let (mut used, mut written, mut flags) = (0u16, 0u16, false);
        for op in blk.ops.iter() {
            let (r, w, f) = uses(op);
            used |= r | w;
            written |= w;
            flags |= f;
        }
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let st = b.block_params(entry)[0];
        let ld = |b: &mut FunctionBuilder<'_>, t: Type, o: usize| b.ins().load(t, MF, st, st_off(o));
        let regs = ld(b, ptr, offset_of!(JitState, regs));
        let sram = ld(b, ptr, offset_of!(JitState, sram));
        let pages = ld(b, ptr, offset_of!(JitState, pages));
        let bank = ld(b, ptr, offset_of!(JitState, bank));
        let budget = ld(b, I64, offset_of!(JitState, budget));
        let r = core::array::from_fn(|_| b.declare_var(I32));
        let f = core::array::from_fn(|_| b.declare_var(I8));
        let acc = b.declare_var(I64);
        let ex = b.declare_var(I8);
        let mut t = Tr { b, ptr, st, regs, sram, pages, bank, core0, sigs, r, f, acc, ex, used, written, flags };
        t.reload();
        let z64 = t.b.ins().iconst(I64, 0);
        t.b.def_var(t.acc, z64);

        let exit = t.b.create_block();
        for ty in [I64, I32, I64, I32] {
            t.b.append_block_param(exit, ty);
        }
        // The caller starts the block with more budget than the plain ops
        // before the first checkpoint take. After each checkpoint the block
        // exits on the exit flag, or when the plain ops up to the next
        // checkpoint could reach the budget: no op after which the
        // interpreter would stop runs unchecked, and stopping early leaves
        // the rest to the caller.
        let runs = super::plain_runs(&blk.ops);
        let z8 = t.b.ins().iconst(I8, 0);
        t.b.def_var(t.ex, z8);
        let n = blk.ops.len();
        for (i, op) in blk.ops.iter().enumerate() {
            let ipc = blk.pc.wrapping_add(op.off as u32);
            let before = t.b.use_var(t.acc);
            let c = t.op(op, i as u32, ipc, before);
            let acc = t.b.ins().iadd(before, c);
            t.b.def_var(t.acc, acc);
            let last = i + 1 == n;
            if last {
                let out = t.b.create_block();
                t.b.ins().jump(out, &[]);
                t.b.switch_to_block(out);
                t.emit_exit(op, i, ipc, acc, c, exit);
            } else if super::plain_cycles(op).is_none() {
                let out = t.b.create_block();
                let next = t.b.create_block();
                let need = t.b.ins().iadd_imm_s(acc, runs[i + 1] as i64);
                let over = t.b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, need, budget);
                let e = t.b.use_var(t.ex);
                let any = t.b.ins().bor(over, e);
                t.b.ins().brif(any, out, &[], next, &[]);
                t.b.switch_to_block(out);
                t.emit_exit(op, i, ipc, acc, c, exit);
                t.b.switch_to_block(next);
            }
        }

        t.b.switch_to_block(exit);
        t.writeback();
        let p: Vec<Value> = t.b.block_params(exit).to_vec();
        let so = |o: usize| st_off(o);
        t.b.ins().store(MF, p[0], st, so(offset_of!(JitState, acc)));
        t.b.ins().store(MF, p[1], st, so(offset_of!(JitState, ops)));
        t.b.ins().store(MF, p[2], st, so(offset_of!(JitState, last_cycles)));
        t.b.ins().store(MF, p[3], st, so(offset_of!(JitState, last_pc)));
        t.b.ins().return_(&[]);
    }

    fn emit_exit(&mut self, op: &Op, i: usize, ipc: u32, acc: Value, c: Value, exit: cranelift_codegen::ir::Block) {
        if !op.k.ends_block() {
            let pc = self.b.ins().iconst(I32, ipc.wrapping_add(op.len()) as i64);
            self.b.ins().store(MF, pc, self.regs, RO + 60);
        }
        let ops = self.b.ins().iconst(I32, i as i64 + 1);
        let ipcv = self.b.ins().iconst(I32, ipc as i64);
        self.b.ins().jump(exit, &[acc.into(), ops.into(), c.into(), ipcv.into()]);
    }

    /// Loads every used register and the flags from the register file.
    fn reload(&mut self) {
        for k in 0..15 {
            if self.used & (1 << k) != 0 {
                let v = self.b.ins().load(I32, MF, self.regs, RO + 4 * k as i32);
                self.b.def_var(self.r[k], v);
            }
        }
        if self.flags {
            let x = self.b.ins().load(I32, MF, self.regs, XO);
            for (j, bit) in [31, 30, 29, 28].into_iter().enumerate() {
                let s = self.b.ins().ushr_imm_u(x, bit);
                let s = self.b.ins().band_imm_u(s, 1);
                let s = self.b.ins().ireduce(I8, s);
                self.b.def_var(self.f[j], s);
            }
        }
    }

    /// Stores every written register and the flags.
    fn writeback(&mut self) {
        for k in 0..15 {
            if self.written & (1 << k) != 0 {
                let v = self.b.use_var(self.r[k]);
                self.b.ins().store(MF, v, self.regs, RO + 4 * k as i32);
            }
        }
        if self.flags {
            let x = self.b.ins().load(I32, MF, self.regs, XO);
            let mut x = self.b.ins().band_imm_u(x, 0x0FFF_FFFF);
            for (j, bit) in [31i64, 30, 29, 28].into_iter().enumerate() {
                let f = self.b.use_var(self.f[j]);
                let f = self.b.ins().uextend(I32, f);
                let f = self.b.ins().ishl_imm_u(f, bit);
                x = self.b.ins().bor(x, f);
            }
            self.b.ins().store(MF, x, self.regs, XO);
        }
    }

    fn get(&mut self, k: u8) -> Value {
        self.b.use_var(self.r[(k & 15) as usize])
    }

    fn set(&mut self, k: u8, v: Value) {
        self.b.def_var(self.r[(k & 15) as usize], v);
    }

    fn k32(&mut self, v: u32) -> Value {
        self.b.ins().iconst(I32, v as i64)
    }

    fn k64(&mut self, v: u64) -> Value {
        self.b.ins().iconst(I64, v as i64)
    }

    fn set_nz(&mut self, res: Value) {
        let n = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, res, 0);
        let z = self.b.ins().icmp_imm_s(IntCC::Equal, res, 0);
        self.b.def_var(self.f[0], n);
        self.b.def_var(self.f[1], z);
    }

    fn set_c(&mut self, c: Value) {
        self.b.def_var(self.f[2], c);
    }

    fn set_v(&mut self, v: Value) {
        self.b.def_var(self.f[3], v);
    }

    /// Low bit of `v` as a flag.
    fn bit(&mut self, v: Value) -> Value {
        let ty = self.b.func.dfg.value_type(v);
        let v = self.b.ins().band_imm_u(v, 1);
        if ty == I8 { v } else { self.b.ins().ireduce(I8, v) }
    }

    /// `a + b`, all four flags.
    fn add(&mut self, a: Value, b: Value) -> Value {
        let res = self.b.ins().iadd(a, b);
        let c = self.b.ins().icmp(IntCC::UnsignedLessThan, res, a);
        let x = self.b.ins().bxor(a, res);
        let y = self.b.ins().bxor(b, res);
        let o = self.b.ins().band(x, y);
        let v = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, o, 0);
        self.set_nz(res);
        self.set_c(c);
        self.set_v(v);
        res
    }

    /// `a - b`, all four flags.
    fn sub(&mut self, a: Value, b: Value) -> Value {
        let res = self.b.ins().isub(a, b);
        let c = self.b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, a, b);
        let x = self.b.ins().bxor(a, b);
        let y = self.b.ins().bxor(a, res);
        let o = self.b.ins().band(x, y);
        let v = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, o, 0);
        self.set_nz(res);
        self.set_c(c);
        self.set_v(v);
        res
    }

    /// `a + b + C`, all four flags.
    fn adc(&mut self, a: Value, b: Value) -> Value {
        let cin = self.b.use_var(self.f[2]);
        let a64 = self.b.ins().uextend(I64, a);
        let b64 = self.b.ins().uextend(I64, b);
        let c64 = self.b.ins().uextend(I64, cin);
        let s = self.b.ins().iadd(a64, b64);
        let s = self.b.ins().iadd(s, c64);
        let res = self.b.ins().ireduce(I32, s);
        let hi = self.b.ins().ushr_imm_u(s, 32);
        let c = self.bit(hi);
        let x = self.b.ins().bxor(a, res);
        let y = self.b.ins().bxor(b, res);
        let o = self.b.ins().band(x, y);
        let v = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, o, 0);
        self.set_nz(res);
        self.set_c(c);
        self.set_v(v);
        res
    }

    fn cond(&mut self, cond: u8) -> Value {
        let [n, z, c, v] = self.f.map(|f| self.b.use_var(f));
        let b = &mut *self.b;
        let not = |b: &mut FunctionBuilder<'_>, x: Value| b.ins().bxor_imm_u(x, 1);
        match cond {
            0 => z,
            1 => not(b, z),
            2 => c,
            3 => not(b, c),
            4 => n,
            5 => not(b, n),
            6 => v,
            7 => not(b, v),
            8 => {
                let nz = not(b, z);
                b.ins().band(c, nz)
            }
            9 => {
                let nc = not(b, c);
                b.ins().bor(nc, z)
            }
            10 => {
                let x = b.ins().bxor(n, v);
                not(b, x)
            }
            11 => b.ins().bxor(n, v),
            12 => {
                let x = b.ins().bxor(n, v);
                let y = b.ins().bor(x, z);
                not(b, y)
            }
            _ => {
                let x = b.ins().bxor(n, v);
                b.ins().bor(x, z)
            }
        }
    }

    // --- memory -------------------------------------------------------------

    /// Fast-path test for an access at `addr`: in SRAM and `bit` clear in
    /// the page byte. Returns (offset, fast).
    fn fast_test(&mut self, addr: Value, bit: i64, slow: cranelift_codegen::ir::Block, fast: cranelift_codegen::ir::Block) -> Value {
        let off = self.b.ins().iadd_imm_s(addr, -0x2000_0000i64);
        let inr = self.b.ins().icmp_imm_u(IntCC::UnsignedLessThan, off, SRAM_SIZE as i64);
        let chk = self.b.create_block();
        self.b.ins().brif(inr, chk, &[], slow, &[]);
        self.b.switch_to_block(chk);
        let pg = self.b.ins().ushr_imm_u(off, 10);
        let pg = self.b.ins().uextend(self.ptr, pg);
        let pa = self.b.ins().iadd(self.pages, pg);
        let pb = self.b.ins().uload8(I32, MF, pa, 0);
        let t = self.b.ins().band_imm_u(pb, bit);
        self.b.ins().brif(t, slow, &[], fast, &[]);
        off
    }

    fn host_addr(&mut self, off: Value) -> Value {
        if self.core0 {
            // bank = off < 0x40000 ? (off >> 2) & 3 : off < 0x41000 ? 4 : 5
            let striped = self.b.ins().icmp_imm_u(IntCC::UnsignedLessThan, off, 0x4_0000);
            let s = self.b.ins().ushr_imm_u(off, 2);
            let s = self.b.ins().band_imm_u(s, 3);
            let lo = self.b.ins().icmp_imm_u(IntCC::UnsignedLessThan, off, 0x4_1000);
            let four = self.k32(4);
            let five = self.k32(5);
            let hi = self.b.ins().select(lo, four, five);
            let bank = self.b.ins().select(striped, s, hi);
            let one = self.k32(1);
            let bitv = self.b.ins().ishl(one, bank);
            let old = self.b.ins().uload8(I32, MF, self.bank, 0);
            let new = self.b.ins().bor(old, bitv);
            self.b.ins().istore8(MF, new, self.bank, 0);
        }
        let o = self.b.ins().uextend(self.ptr, off);
        self.b.ins().iadd(self.sram, o)
    }

    /// Calls [`h_track`] for an access at `addr` when `bit` is set in the
    /// state's `track` word.
    fn track(&mut self, addr: Value, kind: u32, bit: u32) {
        let call = self.b.create_block();
        let done = self.b.create_block();
        let t = self.b.ins().load(I32, MF, self.st, offset_of!(JitState, track) as i32);
        let t = self.b.ins().band_imm_u(t, bit as i64);
        self.b.ins().brif(t, call, &[], done, &[]);
        self.b.switch_to_block(call);
        let k = self.k32(kind);
        let f = self.b.ins().iconst(self.ptr, h_track as *const () as usize as i64);
        self.b.ins().call_indirect(self.sigs[3], f, &[self.st, addr, k]);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
    }

    fn slow_call(&mut self, addr: Value, val: Value, kind: u32, ipc: u32) -> Value {
        let k = self.k32(kind);
        let p = self.k32(ipc);
        let f = self.b.ins().iconst(self.ptr, h_mem as *const () as usize as i64);
        let call = self.b.ins().call_indirect(self.sigs[0], f, &[self.st, addr, val, k, p]);
        let r = self.b.inst_results(call)[0];
        let hi = self.b.ins().ushr_imm_u(r, 32);
        let e = self.bit(hi);
        let old = self.b.use_var(self.ex);
        let e = self.b.ins().bor(old, e);
        self.b.def_var(self.ex, e);
        self.b.ins().ireduce(I32, r)
    }

    /// Load of `size` bytes, zero-extended.
    fn load(&mut self, addr: Value, size: u32, ipc: u32) -> Value {
        let slow = self.b.create_block();
        let fast = self.b.create_block();
        let join = self.b.create_block();
        self.b.append_block_param(join, I32);
        let off = self.fast_test(addr, 1, slow, fast);
        if size == 4 {
            // TIMERAWH and TIMERAWL, any APB alias, from the state.
            let call = self.b.create_block();
            let timer = self.b.create_block();
            let ok = self.b.create_block();
            self.b.switch_to_block(slow);
            let canon = self.b.ins().band_imm_u(addr, !0x3000u32 as i64);
            let reg = self.b.ins().iadd_imm_s(canon, -0x4005_4024i64);
            let is_t = self.b.ins().icmp_imm_u(IntCC::UnsignedLessThan, reg, 8);
            self.b.ins().brif(is_t, timer, &[], call, &[]);
            self.b.switch_to_block(timer);
            let en = self.b.ins().load(I32, MF, self.st, offset_of!(JitState, timer_ok) as i32);
            self.b.ins().brif(en, ok, &[], call, &[]);
            self.b.switch_to_block(ok);
            self.track(addr, size, TRACK_READS);
            // reg is 0 for TIMERAWH, 4 for TIMERAWL.
            let hi = self.b.ins().load(I32, MF, self.st, offset_of!(JitState, timer) as i32 + 4);
            let lo = self.b.ins().load(I32, MF, self.st, offset_of!(JitState, timer) as i32);
            let v = self.b.ins().select(reg, lo, hi);
            self.b.ins().jump(join, &[v.into()]);
            self.b.switch_to_block(call);
            let z = self.k32(0);
            let v = self.slow_call(addr, z, size, ipc);
            self.b.ins().jump(join, &[v.into()]);
            self.b.switch_to_block(fast);
            self.track(addr, size, TRACK_READS);
            let ha = self.host_addr(off);
            let v = self.b.ins().load(I32, MF, ha, 0);
            self.b.ins().jump(join, &[v.into()]);
            self.b.switch_to_block(join);
            return self.b.block_params(join)[0];
        }
        self.b.switch_to_block(fast);
        self.track(addr, size, TRACK_READS);
        let ha = self.host_addr(off);
        let v = match size {
            1 => self.b.ins().uload8(I32, MF, ha, 0),
            2 => self.b.ins().uload16(I32, MF, ha, 0),
            _ => self.b.ins().load(I32, MF, ha, 0),
        };
        self.b.ins().jump(join, &[v.into()]);
        self.b.switch_to_block(slow);
        let z = self.k32(0);
        let v = self.slow_call(addr, z, size, ipc);
        self.b.ins().jump(join, &[v.into()]);
        self.b.switch_to_block(join);
        self.b.block_params(join)[0]
    }

    fn store(&mut self, addr: Value, val: Value, size: u32, ipc: u32) {
        let slow = self.b.create_block();
        let fast = self.b.create_block();
        let join = self.b.create_block();
        let off = self.fast_test(addr, 2, slow, fast);
        self.b.switch_to_block(fast);
        self.track(addr, size | 8, TRACK_STORES);
        let ha = self.host_addr(off);
        match size {
            1 => {
                self.b.ins().istore8(MF, val, ha, 0);
            }
            2 => {
                self.b.ins().istore16(MF, val, ha, 0);
            }
            _ => {
                self.b.ins().store(MF, val, ha, 0);
            }
        }
        self.b.ins().jump(join, &[]);
        self.b.switch_to_block(slow);
        self.slow_call(addr, val, size | 8, ipc);
        self.b.ins().jump(join, &[]);
        self.b.switch_to_block(join);
    }

    /// Branches on `addr`'s alignment to `align`: on a misaligned address
    /// records the fault and yields 1 cycle; otherwise runs `body`, which
    /// returns its cycles.
    fn aligned(&mut self, addr: Value, align: i64, body: impl FnOnce(&mut Self) -> Value) -> Value {
        let bad = self.b.create_block();
        let ok = self.b.create_block();
        let join = self.b.create_block();
        self.b.append_block_param(join, I64);
        let m = self.b.ins().band_imm_u(addr, align - 1);
        self.b.ins().brif(m, bad, &[], ok, &[]);
        self.b.switch_to_block(bad);
        let f = self.b.ins().iconst(self.ptr, h_fault as *const () as usize as i64);
        self.b.ins().call_indirect(self.sigs[2], f, &[self.st]);
        let one = self.b.ins().iconst(I8, 1);
        self.b.def_var(self.ex, one);
        let c1 = self.k64(1);
        self.b.ins().jump(join, &[c1.into()]);
        self.b.switch_to_block(ok);
        let c = body(self);
        self.b.ins().jump(join, &[c.into()]);
        self.b.switch_to_block(join);
        self.b.block_params(join)[0]
    }

    // --- one op ---------------------------------------------------------------

    /// Emits `op`; returns its cycles as an I64 value.
    fn op(&mut self, op: &Op, idx: u32, ipc: u32, acc: Value) -> Value {
        let (d, n, m, imm) = (op.d, op.n, op.m, op.imm);
        let one = |t: &mut Self| t.k64(1);
        match op.k {
            K::LslI => {
                let v = self.get(m);
                if imm == 0 {
                    self.set(d, v);
                    self.set_nz(v);
                } else {
                    let res = self.b.ins().ishl_imm_u(v, imm as i64);
                    let c = self.b.ins().ushr_imm_u(v, 32 - imm as i64);
                    let c = self.bit(c);
                    self.set(d, res);
                    self.set_nz(res);
                    self.set_c(c);
                }
                one(self)
            }
            K::LsrI => {
                let v = self.get(m);
                let (res, c) = if imm == 0 {
                    let z = self.k32(0);
                    let c = self.b.ins().ushr_imm_u(v, 31);
                    (z, c)
                } else {
                    let r = self.b.ins().ushr_imm_u(v, imm as i64);
                    let c = self.b.ins().ushr_imm_u(v, imm as i64 - 1);
                    (r, c)
                };
                let c = self.bit(c);
                self.set(d, res);
                self.set_nz(res);
                self.set_c(c);
                one(self)
            }
            K::AsrI => {
                let v = self.get(m);
                let (res, c) = if imm == 0 {
                    let r = self.b.ins().sshr_imm_u(v, 31);
                    (r, r)
                } else {
                    let r = self.b.ins().sshr_imm_u(v, imm as i64);
                    let c = self.b.ins().sshr_imm_u(v, imm as i64 - 1);
                    (r, c)
                };
                let c = self.bit(c);
                self.set(d, res);
                self.set_nz(res);
                self.set_c(c);
                one(self)
            }
            K::AddR | K::AddI => {
                let a = self.get(n);
                let b = if op.k == K::AddR { self.get(m) } else { self.k32(imm) };
                let r = self.add(a, b);
                self.set(d, r);
                one(self)
            }
            K::SubR | K::SubI => {
                let a = self.get(n);
                let b = if op.k == K::SubR { self.get(m) } else { self.k32(imm) };
                let r = self.sub(a, b);
                self.set(d, r);
                one(self)
            }
            K::MovI => {
                let v = self.k32(imm);
                self.set(d, v);
                self.set_nz(v);
                one(self)
            }
            K::CmpI => {
                let a = self.get(n);
                let b = self.k32(imm);
                self.sub(a, b);
                one(self)
            }
            K::And | K::Eor | K::Orr | K::Bic | K::Mul | K::Tst => {
                let a = self.get(d);
                let b = self.get(m);
                let res = match op.k {
                    K::And | K::Tst => self.b.ins().band(a, b),
                    K::Eor => self.b.ins().bxor(a, b),
                    K::Orr => self.b.ins().bor(a, b),
                    K::Bic => self.b.ins().band_not(a, b),
                    _ => self.b.ins().imul(a, b),
                };
                if op.k != K::Tst {
                    self.set(d, res);
                }
                self.set_nz(res);
                one(self)
            }
            K::Mvn => {
                let a = self.get(m);
                let res = self.b.ins().bnot(a);
                self.set(d, res);
                self.set_nz(res);
                one(self)
            }
            K::LslR | K::LsrR | K::AsrR | K::Ror => {
                let a = self.get(d);
                let s = self.get(m);
                let s = self.b.ins().band_imm_u(s, 0xFF);
                let (res, c) = match op.k {
                    K::LslR => {
                        let k = self.k32(63);
                        let s = self.b.ins().umin(s, k);
                        let s64 = self.b.ins().uextend(I64, s);
                        let a64 = self.b.ins().uextend(I64, a);
                        let w = self.b.ins().ishl(a64, s64);
                        let res = self.b.ins().ireduce(I32, w);
                        let c = self.b.ins().ushr_imm_u(w, 32);
                        (res, c)
                    }
                    K::LsrR => {
                        let k = self.k32(63);
                        let s = self.b.ins().umin(s, k);
                        let s64 = self.b.ins().uextend(I64, s);
                        let a64 = self.b.ins().uextend(I64, a);
                        let a64 = self.b.ins().ishl_imm_u(a64, 1);
                        let w = self.b.ins().ushr(a64, s64);
                        let r = self.b.ins().ushr_imm_u(w, 1);
                        let res = self.b.ins().ireduce(I32, r);
                        (res, w)
                    }
                    K::AsrR => {
                        let k = self.k32(33);
                        let s = self.b.ins().umin(s, k);
                        let s64 = self.b.ins().uextend(I64, s);
                        let a64 = self.b.ins().sextend(I64, a);
                        let a64 = self.b.ins().ishl_imm_u(a64, 1);
                        let w = self.b.ins().sshr(a64, s64);
                        let r = self.b.ins().sshr_imm_u(w, 1);
                        let res = self.b.ins().ireduce(I32, r);
                        (res, w)
                    }
                    _ => {
                        let s5 = self.b.ins().band_imm_u(s, 31);
                        let x = self.b.ins().rotr(a, s5);
                        let c = self.b.ins().ushr_imm_u(x, 31);
                        (x, c)
                    }
                };
                let c = self.bit(c);
                let zero = self.b.ins().icmp_imm_s(IntCC::Equal, s, 0);
                let res = self.b.ins().select(zero, a, res);
                let oldc = self.b.use_var(self.f[2]);
                let c = self.b.ins().select(zero, oldc, c);
                self.set(d, res);
                self.set_nz(res);
                self.set_c(c);
                one(self)
            }
            K::Adc | K::Sbc => {
                let a = self.get(d);
                let b = self.get(m);
                let b = if op.k == K::Sbc { self.b.ins().bnot(b) } else { b };
                let r = self.adc(a, b);
                self.set(d, r);
                one(self)
            }
            K::Neg => {
                let z = self.k32(0);
                let b = self.get(m);
                let r = self.sub(z, b);
                self.set(d, r);
                one(self)
            }
            K::CmpR | K::CmpHi => {
                let a = self.get(n);
                let b = self.get(m);
                self.sub(a, b);
                one(self)
            }
            K::Cmn => {
                let a = self.get(d);
                let b = self.get(m);
                self.add(a, b);
                one(self)
            }
            K::AddHi => {
                let a = self.get(d);
                let b = self.get(m);
                let r = self.b.ins().iadd(a, b);
                self.set(d, r);
                one(self)
            }
            K::MovHi => {
                let v = self.get(m);
                self.set(d, v);
                one(self)
            }
            K::MovK => {
                let v = self.k32(imm);
                self.set(d, v);
                one(self)
            }
            K::AddSpK | K::AddSp | K::SubSp => {
                let sp = self.get(13);
                let r = if op.k == K::SubSp { self.b.ins().iadd_imm_s(sp, -(imm as i64)) } else { self.b.ins().iadd_imm_s(sp, imm as i64) };
                self.set(if op.k == K::AddSpK { d } else { 13 }, r);
                one(self)
            }
            K::Sxth | K::Sxtb | K::Uxth | K::Uxtb | K::Rev | K::Rev16 | K::Revsh => {
                let v = self.get(m);
                let r = match op.k {
                    K::Sxth => {
                        let h = self.b.ins().ireduce(I16, v);
                        self.b.ins().sextend(I32, h)
                    }
                    K::Sxtb => {
                        let h = self.b.ins().ireduce(I8, v);
                        self.b.ins().sextend(I32, h)
                    }
                    K::Uxth => self.b.ins().band_imm_u(v, 0xFFFF),
                    K::Uxtb => self.b.ins().band_imm_u(v, 0xFF),
                    K::Rev => self.b.ins().bswap(v),
                    K::Rev16 => {
                        let a = self.b.ins().ushr_imm_u(v, 8);
                        let a = self.b.ins().band_imm_u(a, 0x00FF_00FF);
                        let b = self.b.ins().ishl_imm_u(v, 8);
                        let b = self.b.ins().band_imm_u(b, 0xFF00_FF00u32 as i64);
                        self.b.ins().bor(a, b)
                    }
                    _ => {
                        let h = self.b.ins().ireduce(I16, v);
                        let h = self.b.ins().bswap(h);
                        self.b.ins().sextend(I32, h)
                    }
                };
                self.set(d, r);
                one(self)
            }
            K::Nop => one(self),
            K::LdrK => {
                let a = self.k32(imm);
                let v = self.load(a, 4, ipc);
                self.set(d, v);
                self.k64(2)
            }
            K::LdrI | K::LdrR | K::LdrhI | K::LdrhR | K::LdrshR => {
                let base = self.get(n);
                let o = if matches!(op.k, K::LdrI | K::LdrhI) { self.k32(imm) } else { self.get(m) };
                let addr = self.b.ins().iadd(base, o);
                let size = if matches!(op.k, K::LdrI | K::LdrR) { 4 } else { 2 };
                let sign = op.k == K::LdrshR;
                self.aligned(addr, size as i64, |t| {
                    let v = t.load(addr, size, ipc);
                    let v = if sign {
                        let h = t.b.ins().ireduce(I16, v);
                        t.b.ins().sextend(I32, h)
                    } else {
                        v
                    };
                    t.set(d, v);
                    t.k64(2)
                })
            }
            K::LdrbI | K::LdrbR | K::LdrsbR => {
                let base = self.get(n);
                let o = if op.k == K::LdrbI { self.k32(imm) } else { self.get(m) };
                let addr = self.b.ins().iadd(base, o);
                let v = self.load(addr, 1, ipc);
                let v = if op.k == K::LdrsbR {
                    let h = self.b.ins().ireduce(I8, v);
                    self.b.ins().sextend(I32, h)
                } else {
                    v
                };
                self.set(d, v);
                self.k64(2)
            }
            K::StrI | K::StrR | K::StrhI | K::StrhR => {
                let base = self.get(n);
                let o = if matches!(op.k, K::StrI | K::StrhI) { self.k32(imm) } else { self.get(m) };
                let addr = self.b.ins().iadd(base, o);
                let size = if matches!(op.k, K::StrI | K::StrR) { 4 } else { 2 };
                let v = self.get(d);
                self.aligned(addr, size as i64, |t| {
                    t.store(addr, v, size, ipc);
                    t.k64(2)
                })
            }
            K::StrbI | K::StrbR => {
                let base = self.get(n);
                let o = if op.k == K::StrbI { self.k32(imm) } else { self.get(m) };
                let addr = self.b.ins().iadd(base, o);
                let v = self.get(d);
                self.store(addr, v, 1, ipc);
                self.k64(2)
            }
            K::Push => {
                let count = imm.count_ones();
                let sp = self.get(13);
                let base = self.b.ins().iadd_imm_s(sp, -(4 * count as i64));
                self.aligned(base, 4, |t| {
                    t.set(13, base);
                    let mut list = imm;
                    let mut k = 0i64;
                    while list != 0 {
                        let reg = list.trailing_zeros() as u8;
                        list &= list - 1;
                        let v = t.get(reg);
                        let a = t.b.ins().iadd_imm_s(base, 4 * k);
                        t.store(a, v, 4, ipc);
                        k += 1;
                    }
                    t.k64(1 + count as u64)
                })
            }
            K::Pop | K::Ldm => {
                let count = imm.count_ones();
                let rn = if op.k == K::Pop { 13 } else { n };
                let base = self.get(rn);
                self.aligned(base, 4, |t| {
                    let mut list = imm;
                    let mut k = 0i64;
                    while list != 0 {
                        let reg = list.trailing_zeros() as u8;
                        list &= list - 1;
                        let a = t.b.ins().iadd_imm_s(base, 4 * k);
                        let v = t.load(a, 4, ipc);
                        t.set(reg, v);
                        k += 1;
                    }
                    if op.k == K::Pop || imm & (1 << (rn & 15)) == 0 {
                        let a = t.b.ins().iadd_imm_s(base, 4 * k);
                        t.set(rn, a);
                    }
                    t.k64(1 + count as u64)
                })
            }
            K::Stm => {
                let count = imm.count_ones();
                let base = self.get(n);
                self.aligned(base, 4, |t| {
                    let mut list = imm;
                    let mut k = 0i64;
                    while list != 0 {
                        let reg = list.trailing_zeros() as u8;
                        list &= list - 1;
                        let v = t.get(reg);
                        let a = t.b.ins().iadd_imm_s(base, 4 * k);
                        t.store(a, v, 4, ipc);
                        k += 1;
                    }
                    let a = t.b.ins().iadd_imm_s(base, 4 * k);
                    t.set(n, a);
                    t.k64(1 + count as u64)
                })
            }
            K::BCond => {
                let taken = self.cond(m);
                let t = self.k32(imm);
                let f = self.k32(ipc.wrapping_add(2));
                let pc = self.b.ins().select(taken, t, f);
                self.b.ins().store(MF, pc, self.regs, RO + 60);
                let c3 = self.k64(3);
                let c1 = self.k64(1);
                self.b.ins().select(taken, c3, c1)
            }
            K::B => {
                let pc = self.k32(imm);
                self.b.ins().store(MF, pc, self.regs, RO + 60);
                self.k64(3)
            }
            K::Bl => {
                let lr = self.k32(ipc.wrapping_add(4) | 1);
                self.set(14, lr);
                let pc = self.k32(imm);
                self.b.ins().store(MF, pc, self.regs, RO + 60);
                self.k64(4)
            }
            K::Interp | K::InterpEnd => {
                self.writeback();
                let i = self.k32(idx);
                let f = self.b.ins().iconst(self.ptr, h_interp as *const () as usize as i64);
                let call = self.b.ins().call_indirect(self.sigs[1], f, &[self.st, i, acc]);
                let r = self.b.inst_results(call)[0];
                self.reload();
                let hi = self.b.ins().ushr_imm_u(r, 32);
                let e = self.bit(hi);
                self.b.def_var(self.ex, e);
                self.b.ins().band_imm_u(r, 0xFFFF_FFFF)
            }
        }
    }
}
