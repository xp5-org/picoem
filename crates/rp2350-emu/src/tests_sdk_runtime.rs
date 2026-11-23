//! Regression tests for behaviours that pico-sdk 2.x firmware depends on:
//!
//! * ARMv8-M acquire/release load/store family `LDAEX*`, `STLEX*`,
//!   `LDA*`, `STL*`, emitted by `hw_claim_*` and the SDK spin locks.
//! * `WFE` wakes on a pending + enabled interrupt, not only the event flag.
//! * `LDM`/`POP` that loads PC with an EXC_RETURN value must write the base
//!   register back *before* unstacking the exception frame.
//! * D-register memory transfers, `VPUSH`/`VPOP`/`VLDR`/`VSTR` on `Dn`, are
//!   legal on an SP-only FPU; double-precision data processing is not.
//!
//! CPS I/F bit order and TICKS-follows-clk_ref have their regression tests
//! next to the existing CPS tests in `tests.rs` and in
//! `peripherals/phase1_tests.rs`.

use crate::bus::Bus;
use crate::core::CortexM33;
use crate::{Config, Emulator};

fn core_and_bus() -> (CortexM33, Bus) {
    let bus = Bus::new();
    let core = CortexM33::new(0, bus.atomics.clone());
    (core, bus)
}

// ---------------------------------------------------------------------------
// Acquire / release family
// ---------------------------------------------------------------------------

/// `LDAEXB r0,[r1]` / `STLEXB r2,r3,[r1]`: the exact opcodes pico-sdk emits
/// in `hw_claim_or_assert`, in the `e8d4 6fcf` / `e8c4 0f46` style.
#[test]
fn ldaexb_stlexb_roundtrip() {
    let (mut c, mut bus) = core_and_bus();
    c.set_reg(1, 0x2000_0020);
    bus.write8(0x2000_0020, 0x42, 0);
    // LDAEXB r0,[r1]: hw1[11:4] = 0xFC, hw1[3:0] = 0xF.
    let pc0 = c.regs.pc();
    c.execute_one_wide_with_bus(0xE8D1, 0x0FCF, &mut bus);
    assert_eq!(c.regs.pc(), pc0.wrapping_add(4), "must not fall through to LDRD (rt2=PC)");
    assert_eq!(c.reg(0), 0x42);
    assert!(!c.has_pending_fault(), "LDAEXB must not be UNDEFINED");

    c.set_reg(3, 0x99);
    c.set_reg(2, 0xFFFF_FFFF); // status must be *written* as 0, not left at 0
    // STLEXB r2,r3,[r1]: Rt=3, hw1[11:4]=0xFC, Rd=2.
    c.execute_one_wide_with_bus(0xE8C1, (3 << 12) | 0x0FC0 | 2, &mut bus);
    assert_eq!(c.reg(2), 0, "store-exclusive succeeds after matching load");
    assert_eq!(bus.read8(0x2000_0020, 0), 0x99);
}

#[test]
fn ldaexh_stlexh_roundtrip() {
    let (mut c, mut bus) = core_and_bus();
    c.set_reg(1, 0x2000_0030);
    bus.write16(0x2000_0030, 0xABCD, 0);
    let pc0 = c.regs.pc();
    c.execute_one_wide_with_bus(0xE8D1, 0x0FDF, &mut bus);
    assert_eq!(c.regs.pc(), pc0.wrapping_add(4), "must not fall through to LDRD (rt2=PC)"); // LDAEXH r0,[r1]
    assert_eq!(c.reg(0), 0xABCD);
    c.set_reg(3, 0x1234);
    c.set_reg(2, 0xFFFF_FFFF);
    c.execute_one_wide_with_bus(0xE8C1, (3 << 12) | 0x0FD0 | 2, &mut bus); // STLEXH
    assert_eq!(c.reg(2), 0);
    assert_eq!(bus.read16(0x2000_0030, 0), 0x1234);
}

#[test]
fn ldaex_stlex_word_roundtrip() {
    let (mut c, mut bus) = core_and_bus();
    c.set_reg(1, 0x2000_0040);
    bus.write32(0x2000_0040, 0xDEAD_BEEF, 0);
    let pc0 = c.regs.pc();
    c.execute_one_wide_with_bus(0xE8D1, 0x0FEF, &mut bus);
    assert_eq!(c.regs.pc(), pc0.wrapping_add(4), "must not fall through to LDRD (rt2=PC)"); // LDAEX r0,[r1]
    assert_eq!(c.reg(0), 0xDEAD_BEEF);
    c.set_reg(3, 0x1122_3344);
    c.set_reg(2, 0xFFFF_FFFF);
    c.execute_one_wide_with_bus(0xE8C1, (3 << 12) | 0x0FE0 | 2, &mut bus); // STLEX
    assert_eq!(c.reg(2), 0);
    assert_eq!(bus.read32(0x2000_0040, 0), 0x1122_3344);
}

/// A store-exclusive without a preceding load-exclusive must fail and leave
/// memory untouched.
#[test]
fn stlexb_without_monitor_fails() {
    let (mut c, mut bus) = core_and_bus();
    c.set_reg(1, 0x2000_0050);
    bus.write8(0x2000_0050, 0x11, 0);
    c.set_reg(3, 0x77);
    c.execute_one_wide_with_bus(0xE8C1, (3 << 12) | 0x0FC0 | 2, &mut bus);
    assert_eq!(c.reg(2), 1, "monitor open -> status 1");
    assert_eq!(bus.read8(0x2000_0050, 0), 0x11);
}

/// Plain acquire/release `LDAB`/`LDAH`/`LDA`, `STLB`/`STLH`/`STL`, with no
/// monitor involved. `stlb ip,[r3]` = `e8c3 cf8f` is the SDK's lock release.
#[test]
fn lda_stl_plain_variants() {
    let (mut c, mut bus) = core_and_bus();
    c.set_reg(1, 0x2000_0060);
    bus.write32(0x2000_0060, 0xA1B2_C3D4, 0);

    c.execute_one_wide_with_bus(0xE8D1, 0x0F8F, &mut bus); // LDAB r0,[r1]
    assert_eq!(c.reg(0), 0xD4);
    c.execute_one_wide_with_bus(0xE8D1, 0x0F9F, &mut bus); // LDAH r0,[r1]
    assert_eq!(c.reg(0), 0xC3D4);
    c.execute_one_wide_with_bus(0xE8D1, 0x0FAF, &mut bus); // LDA r0,[r1]
    assert_eq!(c.reg(0), 0xA1B2_C3D4);

    c.set_reg(2, 0x5566_7788);
    c.execute_one_wide_with_bus(0xE8C1, (2 << 12) | 0x0F8F, &mut bus); // STLB r2,[r1]
    assert_eq!(bus.read32(0x2000_0060, 0), 0xA1B2_C388);
    c.execute_one_wide_with_bus(0xE8C1, (2 << 12) | 0x0F9F, &mut bus); // STLH
    assert_eq!(bus.read32(0x2000_0060, 0), 0xA1B2_7788);
    c.execute_one_wide_with_bus(0xE8C1, (2 << 12) | 0x0FAF, &mut bus); // STL
    assert_eq!(bus.read32(0x2000_0060, 0), 0x5566_7788);
}

// ---------------------------------------------------------------------------
// WFE wake on pending + enabled interrupt
// ---------------------------------------------------------------------------

/// pico-sdk `sleep_ms` ends in `wfe` and relies on the alarm IRQ, not an
/// explicit SEV, to wake the core.
#[test]
fn wfe_wakes_on_pending_enabled_irq() {
    let mut emu = Emulator::new(Config::default());
    emu.bus.atomics.set_halted(1);
    emu.bus.atomics.set_halted(0); // keep the pair from executing code
    emu.bus.atomics.set_wfe_waiting(0);
    emu.mmio_write32(0xE000_E100, 1 << 3); // NVIC_ISER0: enable IRQ 3
    emu.mmio_write32(0xE000_E200, 1 << 3); // NVIC_ISPR0: pend IRQ 3
    let _ = emu.step().unwrap();
    assert!(
        !emu.bus.atomics.is_wfe_waiting(0),
        "a pending, enabled interrupt must end WFE sleep"
    );
}

/// Negative control: pending but *disabled* must not wake WFE.
#[test]
fn wfe_stays_asleep_when_pending_irq_is_disabled() {
    let mut emu = Emulator::new(Config::default());
    emu.bus.atomics.set_halted(1);
    emu.bus.atomics.set_halted(0);
    emu.bus.atomics.set_wfe_waiting(0);
    emu.mmio_write32(0xE000_E200, 1 << 3); // pend IRQ 3, but never enable it
    let _ = emu.step().unwrap();
    assert!(emu.bus.atomics.is_wfe_waiting(0));
}

// ---------------------------------------------------------------------------
// LDM / POP {..., pc} with EXC_RETURN
// ---------------------------------------------------------------------------

/// Handler epilogue `push {r4,lr}` ... `ldmia sp!,{r4,pc}` with
/// LR = EXC_RETURN. The exception return must see the post-writeback SP;
/// unstacking from SP-8 would load a garbage frame.
#[test]
fn ldmia_pc_exc_return_writes_back_sp_before_unstacking() {
    let (mut c, mut bus) = core_and_bus();
    c.regs.msp = 0x2000_2000;
    c.regs.r[13] = c.regs.msp;
    let msp_pre_entry = c.regs.msp;
    let vtor = 0x2000_4000;
    c.ppb.vtor = vtor;
    bus.write32(vtor + 14 * 4, 0x2000_0200 | 1, 0);

    c.test_enter_exception(14, &mut bus);
    assert_eq!(c.regs.ipsr(), 14);
    let sp_in_handler = c.regs.r[13];

    c.regs.r[14] = 0xFFFF_FFF9; // EXC_RETURN: thread, MSP, basic frame
    c.set_reg(4, 0x4444_4444);
    // STMDB sp!,{r4,lr} = push {r4,lr} = e92d 4010
    c.execute_one_wide_with_bus(0xE92D, 0x4010, &mut bus);
    assert_eq!(c.regs.r[13], sp_in_handler - 8);
    c.set_reg(4, 0);
    // LDMIA sp!,{r4,pc} = pop {r4,pc} = e8bd 8010
    c.execute_one_wide_with_bus(0xE8BD, 0x8010, &mut bus);

    assert_eq!(c.regs.ipsr(), 0, "returned to thread mode");
    assert_eq!(c.reg(4), 0x4444_4444, "r4 restored from the handler's push");
    assert_eq!(c.regs.msp, msp_pre_entry, "exception frame fully unstacked");
}

// ---------------------------------------------------------------------------
// D-register transfers on an SP-only FPU
// ---------------------------------------------------------------------------

/// `vpush {d8}` / `vpop {d8}`: the SDK's IRQ-handler prologue/epilogue,
/// `ed2d 8b02` / `ecbd 8b02`. Dn maps to S[2n], S[2n+1].
#[test]
fn vpush_vpop_d8_roundtrip() {
    let (mut c, mut bus) = core_and_bus();
    c.regs.r[13] = 0x2000_0100;
    c.regs.s[16] = f32::from_bits(0x1111_1111);
    c.regs.s[17] = f32::from_bits(0x2222_2222);

    c.execute_one_wide_with_bus(0xED2D, 0x8B02, &mut bus); // VPUSH {d8}
    assert!(!c.has_pending_fault(), "VPUSH {{d8}} must not be UNDEFINED");
    assert_eq!(c.regs.r[13], 0x2000_00F8);
    assert_eq!(bus.read32(0x2000_00F8, 0), 0x1111_1111);
    assert_eq!(bus.read32(0x2000_00FC, 0), 0x2222_2222);

    c.regs.s[16] = 0.0;
    c.regs.s[17] = 0.0;
    c.execute_one_wide_with_bus(0xECBD, 0x8B02, &mut bus); // VPOP {d8}
    assert_eq!(c.regs.r[13], 0x2000_0100);
    assert_eq!(c.regs.s[16].to_bits(), 0x1111_1111);
    assert_eq!(c.regs.s[17].to_bits(), 0x2222_2222);
}

#[test]
fn vstr_vldr_d8_64bit() {
    let (mut c, mut bus) = core_and_bus();
    c.set_reg(1, 0x2000_0200);
    c.regs.s[16] = f32::from_bits(0xAAAA_0001);
    c.regs.s[17] = f32::from_bits(0xBBBB_0002);
    c.execute_one_wide_with_bus(0xED81, 0x8B02, &mut bus); // VSTR d8,[r1,#8]
    assert_eq!(bus.read32(0x2000_0208, 0), 0xAAAA_0001);
    assert_eq!(bus.read32(0x2000_020C, 0), 0xBBBB_0002);

    c.regs.s[16] = 0.0;
    c.regs.s[17] = 0.0;
    c.execute_one_wide_with_bus(0xED91, 0x8B02, &mut bus); // VLDR d8,[r1,#8]
    assert_eq!(c.regs.s[16].to_bits(), 0xAAAA_0001);
    assert_eq!(c.regs.s[17].to_bits(), 0xBBBB_0002);
}

/// Only the memory transfers are legal: double-precision *data processing*,
/// here `vadd.f64 d0,d1,d2` = `ee31 0b02`, is still UNDEFINED on RP2350.
#[test]
fn double_precision_data_processing_still_undefined() {
    let (mut c, mut bus) = core_and_bus();
    c.execute_one_wide_with_bus(0xEE31, 0x0B02, &mut bus);
    assert!(c.has_pending_fault(), "vadd.f64 must remain UNDEFINED");
}
