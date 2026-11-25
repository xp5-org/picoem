// Underscore positions inside binary literals document RISC-V
// instruction-encoding bit-fields, not 4-bit visual groups.
#![allow(clippy::unusual_byte_groupings)]

// Zba, Zbb, Zbs and Zbkb semantic tests. Every case goes through the full
// fetch, decode and execute path: the word is planted at the reset PC,
// x1 and x2 hold the operands, and the result lands in x3. Golden values
// follow the RISC-V bit-manipulation spec v1.0.0 and the scalar crypto
// spec v1.0.1 for Zbkb.

use super::decode::{self, Op};
use super::tests_common::{fresh, write_insn};

const OP: u32 = 0b01_100;
const OP_IMM: u32 = 0b00_100;

fn enc_r(f7: u32, f3: u32) -> u32 {
    (f7 << 25) | (2 << 20) | (1 << 15) | (f3 << 12) | (3 << 7) | (OP << 2) | 0b11
}

fn enc_i(imm12: u32, f3: u32) -> u32 {
    (imm12 << 20) | (1 << 15) | (f3 << 12) | (3 << 7) | (OP_IMM << 2) | 0b11
}

/// Step one instruction with x1 = a and x2 = b; return x3. Asserts the
/// instruction retired without a trap, so the PC moved to the next word.
fn run(insn: u32, a: u32, b: u32) -> u32 {
    let (mut c, mut bus) = fresh();
    write_insn(&mut bus, 0, insn);
    c.x[1] = a;
    c.x[2] = b;
    c.step(&mut bus);
    assert_eq!(c.pc, 0x2000_0004, "insn {insn:#010x} trapped or did not retire");
    c.x[3]
}

// ---------- Zba ----------

#[test]
fn zba_shnadd() {
    assert_eq!(run(enc_r(0b001_0000, 0b010), 3, 10), 16);
    assert_eq!(run(enc_r(0b001_0000, 0b100), 3, 10), 22);
    assert_eq!(run(enc_r(0b001_0000, 0b110), 3, 10), 34);
    assert_eq!(run(enc_r(0b001_0000, 0b110), 0x8000_0001, 0), 8);
}

// ---------- Zbb ----------

#[test]
fn zbb_logic_with_negate() {
    assert_eq!(run(enc_r(0b010_0000, 0b111), 0xFF00_FF00, 0x0F0F_0F0F), 0xF000_F000);
    assert_eq!(run(enc_r(0b010_0000, 0b110), 0x0000_00F0, 0xFFFF_FF0F), 0x0000_00F0);
    assert_eq!(run(enc_r(0b010_0000, 0b100), 0xFFFF_0000, 0xFF00_FF00), 0xFF00_00FF);
}

#[test]
fn zbb_count() {
    assert_eq!(run(enc_i(0x600, 0b001), 0x0001_0000, 0), 15);
    assert_eq!(run(enc_i(0x600, 0b001), 0, 0), 32);
    assert_eq!(run(enc_i(0x601, 0b001), 0x0001_0000, 0), 16);
    assert_eq!(run(enc_i(0x601, 0b001), 0, 0), 32);
    assert_eq!(run(enc_i(0x602, 0b001), 0xF0F0_0001, 0), 9);
}

#[test]
fn zbb_min_max() {
    assert_eq!(run(enc_r(0b000_0101, 0b110), 0xFFFF_FFFF, 1), 1);
    assert_eq!(run(enc_r(0b000_0101, 0b111), 0xFFFF_FFFF, 1), 0xFFFF_FFFF);
    assert_eq!(run(enc_r(0b000_0101, 0b100), 0xFFFF_FFFF, 1), 0xFFFF_FFFF);
    assert_eq!(run(enc_r(0b000_0101, 0b101), 0xFFFF_FFFF, 1), 1);
}

#[test]
fn zbb_extend() {
    assert_eq!(run(enc_i(0x604, 0b001), 0x0000_0080, 0), 0xFFFF_FF80);
    assert_eq!(run(enc_i(0x604, 0b001), 0xFFFF_FF7F, 0), 0x0000_007F);
    assert_eq!(run(enc_i(0x605, 0b001), 0x1234_8000, 0), 0xFFFF_8000);
    // zext.h is pack with rs2 = x0.
    let zext_h = (0b000_0100 << 25) | (1 << 15) | (0b100 << 12) | (3 << 7) | (OP << 2) | 0b11;
    assert_eq!(run(zext_h, 0xABCD_1234, 0xFFFF_FFFF), 0x0000_1234);
}

#[test]
fn zbb_rotate() {
    assert_eq!(run(enc_r(0b011_0000, 0b001), 0x8000_0001, 1), 0x0000_0003);
    assert_eq!(run(enc_r(0b011_0000, 0b001), 0x8000_0001, 33), 0x0000_0003);
    assert_eq!(run(enc_r(0b011_0000, 0b101), 0x8000_0001, 1), 0xC000_0000);
    assert_eq!(run(enc_i(0x600 | 4, 0b101), 0x1234_5678, 0), 0x8123_4567);
}

#[test]
fn zbb_byte_ops() {
    assert_eq!(run(enc_i(0x287, 0b101), 0x0010_0001, 0), 0x00FF_00FF);
    assert_eq!(run(enc_i(0x698, 0b101), 0x1234_5678, 0), 0x7856_3412);
}

// ---------- Zbs ----------

#[test]
fn zbs_register_forms() {
    assert_eq!(run(enc_r(0b010_0100, 0b001), 0xFFFF_FFFF, 31), 0x7FFF_FFFF);
    assert_eq!(run(enc_r(0b010_0100, 0b101), 0x0000_0400, 10), 1);
    assert_eq!(run(enc_r(0b010_0100, 0b101), 0x0000_0400, 11), 0);
    assert_eq!(run(enc_r(0b011_0100, 0b001), 0, 35), 8);
    assert_eq!(run(enc_r(0b001_0100, 0b001), 0, 0), 1);
}

#[test]
fn zbs_immediate_forms() {
    assert_eq!(run(enc_i(0x480 | 31, 0b001), 0xFFFF_FFFF, 0), 0x7FFF_FFFF);
    assert_eq!(run(enc_i(0x480 | 10, 0b101), 0x0000_0400, 0), 1);
    assert_eq!(run(enc_i(0x680 | 3, 0b001), 0x0000_0008, 0), 0);
    assert_eq!(run(enc_i(0x280, 0b001), 0, 0), 1);
}

// ---------- Zbkb ----------

#[test]
fn zbkb_pack() {
    assert_eq!(run(enc_r(0b000_0100, 0b100), 0xAAAA_1234, 0xBBBB_5678), 0x5678_1234);
    assert_eq!(run(enc_r(0b000_0100, 0b111), 0xFFFF_FF12, 0xFFFF_FF34), 0x0000_3412);
}

#[test]
fn zbkb_brev8() {
    assert_eq!(run(enc_i(0x687, 0b101), 0x0102_0480, 0), 0x8040_2001);
}

#[test]
fn zbkb_zip_unzip() {
    assert_eq!(run(enc_i(0x08F, 0b001), 0x0000_FFFF, 0), 0x5555_5555);
    assert_eq!(run(enc_i(0x08F, 0b001), 0xFFFF_0000, 0), 0xAAAA_AAAA);
    assert_eq!(run(enc_i(0x08F, 0b101), 0x5555_5555, 0), 0x0000_FFFF);
    assert_eq!(run(enc_i(0x08F, 0b101), 0xAAAA_AAAA, 0), 0xFFFF_0000);
    let x = 0x9E37_79B9;
    let z = run(enc_i(0x08F, 0b001), x, 0);
    assert_eq!(run(enc_i(0x08F, 0b101), z, 0), x);
}

// ---------- encodings that stay illegal ----------

#[test]
fn neighbours_stay_illegal() {
    let illegal = [
        // Zbc clmul, clmulr, clmulh: not in the RP2350 Hazard3 configuration.
        enc_r(0b000_0101, 0b001),
        enc_r(0b000_0101, 0b010),
        enc_r(0b000_0101, 0b011),
        // Unused unary slot next to sext.h.
        enc_i(0x603, 0b001),
        // rev8 RV64 encoding.
        enc_i(0x6B8, 0b101),
        // Zbkx xperm8: not implemented.
        enc_r(0b001_0100, 0b100),
        // rori with shamt bit 5 set is RV64 only.
        enc_i(0x620, 0b101),
    ];
    for insn in illegal {
        assert!(
            matches!(decode::decode(insn), Op::Illegal { .. }),
            "{insn:#010x} should decode illegal"
        );
    }
}
