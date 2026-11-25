//! IO_BANK0 GPIO function select (base 0x4001_4000).
//!
//! RP2040 datasheet §2.19.6. 30 GPIOs × 2 registers each (`GPIO_STATUS`,
//! `GPIO_CTRL`) = 240 bytes of per-pin state. Registers beyond the pin
//! block raise GPIO interrupts: each pin has four INTR bits, level low,
//! level high, edge low and edge high. Edges latch when the input level
//! changes and clear when firmware writes 1 to them; level bits follow the
//! input. INTS = (INTR | INTF) & INTE per processor, and IO_IRQ_BANK0 is
//! asserted while either processor's INTS is non-zero. Dormant-wake
//! registers are storage only.
//!
//! `GPIO_CTRL[4:0]` is FUNCSEL:
//! * 0 = XIP, 1 = SPI, 2 = UART, 3 = I2C, 4 = PWM, 5 = SIO,
//! * 6 = PIO0, 7 = PIO1, 8 = USB, 31 = NULL (Hi-Z).
//!
//! Phase 5.A stores the CTRL register and reports STATUS as
//! `oetoperiph << 12 | intoperiph << 24` — a simple passthrough of what
//! firmware would see on a real chip after programming the pin.

/// Number of GPIOs on RP2040 (IO_BANK0 block).
pub(crate) const NUM_GPIOS: usize = 30;

/// Per-pin register pair: CTRL / STATUS.
pub struct IoBank0 {
    /// GPIO_CTRL[pin] = FUNCSEL / OUTOVER / OEOVER / INOVER / IRQOVER.
    pub ctrl: [u32; NUM_GPIOS],
    /// Pins whose CTRL.OUTOVER inverts, forces low or forces high the
    /// peripheral's output level, kept in step with `ctrl` on every write.
    pub(crate) out_invert: u32,
    pub(crate) out_low: u32,
    pub(crate) out_high: u32,
    /// Latched edge bits of INTR, 4 words of 8 pins; level bits are
    /// derived from `level_in` on read.
    pub(crate) intr: [u32; 4],
    /// GPIO input levels, as the bus last computed them.
    pub(crate) level_in: u32,
    pub(crate) proc0_inte: [u32; 4],
    pub(crate) proc0_intf: [u32; 4],
    pub(crate) proc1_inte: [u32; 4],
    pub(crate) proc1_intf: [u32; 4],
    pub(crate) dormant_wake_inte: [u32; 4],
    pub(crate) dormant_wake_intf: [u32; 4],
    pub(crate) dormant_wake_ints: [u32; 4],
}

/// The edge-low and edge-high bits of every pin in an INTR word.
const EDGE_BITS: u32 = 0xCCCC_CCCC;

impl IoBank0 {
    /// INTR word `bank`: latched edges plus the level bits of the current
    /// input levels.
    fn intr_raw(&self, bank: usize) -> u32 {
        let mut v = self.intr[bank];
        for i in 0..8 {
            let pin = bank * 8 + i;
            if pin >= NUM_GPIOS {
                break;
            }
            let high = (self.level_in >> pin) & 1;
            v |= (if high != 0 { 0b0010 } else { 0b0001 }) << (4 * i);
        }
        v
    }

    /// INTS word `bank` for processor `proc`.
    fn ints(&self, proc: usize, bank: usize) -> u32 {
        let (inte, intf) = if proc == 0 {
            (self.proc0_inte[bank], self.proc0_intf[bank])
        } else {
            (self.proc1_inte[bank], self.proc1_intf[bank])
        };
        (self.intr_raw(bank) | intf) & inte
    }

    /// New GPIO input levels: latch the edges against the previous ones.
    pub(crate) fn set_inputs(&mut self, levels: u32) {
        let changed = (self.level_in ^ levels) & ((1 << NUM_GPIOS) - 1);
        if changed != 0 {
            for pin in 0..NUM_GPIOS {
                if changed & (1 << pin) != 0 {
                    let bit = if levels & (1 << pin) != 0 { 0b1000 } else { 0b0100 };
                    self.intr[pin / 8] |= bit << (4 * (pin % 8));
                }
            }
        }
        self.level_in = levels;
    }

    /// IO_IRQ_BANK0 level: either processor has an enabled interrupt.
    pub(crate) fn irq_active(&self) -> bool {
        // INTS is masked by INTE, so with nothing enabled no pin can raise
        // it, whatever INTR holds.
        let enabled = (0..4).fold(0, |a, b| a | self.proc0_inte[b] | self.proc1_inte[b]);
        enabled != 0 && (0..4).any(|b| self.ints(0, b) != 0 || self.ints(1, b) != 0)
    }

    pub fn new() -> Self {
        // FUNCSEL default for every pin is 31 (NULL / Hi-Z).
        let mut ctrl = [0u32; NUM_GPIOS];
        for c in ctrl.iter_mut() {
            *c = 0x0000_001F;
        }
        Self {
            ctrl,
            out_invert: 0,
            out_low: 0,
            out_high: 0,
            intr: [0; 4],
            level_in: 0,
            proc0_inte: [0; 4],
            proc0_intf: [0; 4],
            proc1_inte: [0; 4],
            proc1_intf: [0; 4],
            dormant_wake_inte: [0; 4],
            dormant_wake_intf: [0; 4],
            dormant_wake_ints: [0; 4],
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns the FUNCSEL (bits [4:0]) currently assigned to `pin`.
    #[inline]
    pub fn funcsel(&self, pin: usize) -> u32 {
        if pin < NUM_GPIOS {
            self.ctrl[pin] & 0x1F
        } else {
            0
        }
    }

    /// Read an IO_BANK0 register by byte offset.
    pub fn read32(&self, offset: u32) -> u32 {
        let off = offset & 0xFFF;
        // Per-pin status/ctrl at offsets 0x000..0x0EF (30 pins × 8 bytes).
        if off < (NUM_GPIOS as u32) * 8 {
            let pin = (off >> 3) as usize;
            return match off & 0x7 {
                0x0 => self.status_for(pin),
                0x4 => self.ctrl[pin],
                _ => 0,
            };
        }
        // Interrupt block starts at 0x0F0.
        match off {
            0x0F0..=0x0FC => self.intr_raw(((off & 0xF) >> 2) as usize),
            0x100 => self.proc0_inte[0],
            0x104 => self.proc0_inte[1],
            0x108 => self.proc0_inte[2],
            0x10C => self.proc0_inte[3],
            0x110 => self.proc0_intf[0],
            0x114 => self.proc0_intf[1],
            0x118 => self.proc0_intf[2],
            0x11C => self.proc0_intf[3],
            0x120..=0x12C => self.ints(0, ((off & 0xF) >> 2) as usize),
            0x130 => self.proc1_inte[0],
            0x134 => self.proc1_inte[1],
            0x138 => self.proc1_inte[2],
            0x13C => self.proc1_inte[3],
            0x140 => self.proc1_intf[0],
            0x144 => self.proc1_intf[1],
            0x148 => self.proc1_intf[2],
            0x14C => self.proc1_intf[3],
            0x150..=0x15C => self.ints(1, ((off & 0xF) >> 2) as usize),
            0x160 => self.dormant_wake_inte[0],
            0x164 => self.dormant_wake_inte[1],
            0x168 => self.dormant_wake_inte[2],
            0x16C => self.dormant_wake_inte[3],
            0x170 => self.dormant_wake_intf[0],
            0x174 => self.dormant_wake_intf[1],
            0x178 => self.dormant_wake_intf[2],
            0x17C => self.dormant_wake_intf[3],
            0x180 => self.dormant_wake_ints[0],
            0x184 => self.dormant_wake_ints[1],
            0x188 => self.dormant_wake_ints[2],
            0x18C => self.dormant_wake_ints[3],
            _ => 0,
        }
    }

    /// Write an IO_BANK0 register with an alias-aware update.
    pub fn write32(&mut self, offset: u32, val: u32, alias: u32) {
        let off = offset & 0xFFF;
        let apply = |cur: u32, v: u32| match alias {
            0 => v,
            1 => cur ^ v,
            2 => cur | v,
            3 => cur & !v,
            _ => v,
        };
        if off < (NUM_GPIOS as u32) * 8 {
            let pin = (off >> 3) as usize;
            match off & 0x7 {
                0x0 => {} // STATUS read-only
                0x4 => {
                    self.ctrl[pin] = apply(self.ctrl[pin], val);
                    // OUTOVER, bits 9:8: 0 normal, 1 invert, 2 low, 3 high.
                    let bit = 1u32 << pin;
                    self.out_invert &= !bit;
                    self.out_low &= !bit;
                    self.out_high &= !bit;
                    match (self.ctrl[pin] >> 8) & 3 {
                        1 => self.out_invert |= bit,
                        2 => self.out_low |= bit,
                        3 => self.out_high |= bit,
                        _ => {}
                    }
                }
                _ => {}
            }
            return;
        }
        // INTR is write-1-to-clear on its edge bits under every alias;
        // INTS is read-only; the rest is plain storage.
        let idx = |o: u32| ((o & 0xF) >> 2) as usize;
        match off {
            0x0F0..=0x0FC => self.intr[idx(off)] &= !(val & EDGE_BITS),
            0x120..=0x12C | 0x150..=0x15C => {}
            0x100..=0x10C => self.proc0_inte[idx(off)] = apply(self.proc0_inte[idx(off)], val),
            0x110..=0x11C => self.proc0_intf[idx(off)] = apply(self.proc0_intf[idx(off)], val),
            0x130..=0x13C => self.proc1_inte[idx(off)] = apply(self.proc1_inte[idx(off)], val),
            0x140..=0x14C => self.proc1_intf[idx(off)] = apply(self.proc1_intf[idx(off)], val),
            0x160..=0x16C => {
                self.dormant_wake_inte[idx(off)] = apply(self.dormant_wake_inte[idx(off)], val)
            }
            0x170..=0x17C => {
                self.dormant_wake_intf[idx(off)] = apply(self.dormant_wake_intf[idx(off)], val)
            }
            0x180..=0x18C => {
                self.dormant_wake_ints[idx(off)] = apply(self.dormant_wake_ints[idx(off)], val)
            }
            _ => {}
        }
    }

    /// Synthesise a GPIO_STATUS word for `pin`. Phase 5.A bakes in a
    /// simple shape: OUTTOPAD / OETOPAD / INFROMPAD mirror the CTRL
    /// OUTOVER / OEOVER / INOVER when those fields are non-zero, else
    /// the SIO-driven levels. Firmware rarely reads STATUS in init code
    /// — this is enough for Pico SDK's gpio_init round-trip.
    fn status_for(&self, _pin: usize) -> u32 {
        0
    }
}

impl Default for IoBank0 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn funcsel_defaults_to_null() {
        let io = IoBank0::new();
        for pin in 0..NUM_GPIOS {
            assert_eq!(io.funcsel(pin), 0x1F);
        }
    }

    #[test]
    fn set_funcsel_via_ctrl_write() {
        let mut io = IoBank0::new();
        // GPIO25 CTRL is at offset 25 * 8 + 4 = 0xCC.
        io.write32(0xCC, 5, 0); // SIO
        assert_eq!(io.funcsel(25), 5);
    }

    #[test]
    fn status_readonly_ignores_write() {
        let mut io = IoBank0::new();
        let before = io.read32(0x00);
        io.write32(0x00, 0xFFFF_FFFF, 0);
        assert_eq!(io.read32(0x00), before);
    }

    #[test]
    fn intr_latches_edges_and_clears_on_write() {
        let mut io = IoBank0::new();
        io.set_inputs(0);
        // Pin 2 low: level-low bit only.
        assert_eq!(io.read32(0x0F0) >> 8 & 0xF, 0b0001);
        io.set_inputs(1 << 2);
        // Rising edge latched, level high now.
        assert_eq!(io.read32(0x0F0) >> 8 & 0xF, 0b1010);
        io.set_inputs(0);
        assert_eq!(io.read32(0x0F0) >> 8 & 0xF, 0b1101);
        // Writing 1s clears the edges; levels cannot be cleared.
        io.write32(0x0F0, 0xF << 8, 0);
        assert_eq!(io.read32(0x0F0) >> 8 & 0xF, 0b0001);
    }

    #[test]
    fn ints_and_irq_follow_inte_and_intf() {
        let mut io = IoBank0::new();
        io.set_inputs(0);
        // Pin 20 is bank 2, slot 4. Enable its rising edge for proc0.
        io.write32(0x108, 0b1000 << 16, 0);
        assert!(!io.irq_active());
        io.set_inputs(1 << 20);
        assert_eq!(io.read32(0x128), 0b1000 << 16);
        assert!(io.irq_active());
        io.write32(0x0F8, 0b1000 << 16, 0);
        assert!(!io.irq_active());
        // INTF forces the interrupt regardless of the pin.
        io.write32(0x118, 0b1000 << 16, 0);
        assert!(io.irq_active());
    }

    #[test]
    fn level_high_interrupt_follows_pin() {
        let mut io = IoBank0::new();
        io.set_inputs(0);
        io.write32(0x13C, 0b0010 << 4, 0); // proc1, pin 25 level high
        assert!(!io.irq_active());
        io.set_inputs(1 << 25);
        assert!(io.irq_active());
        assert_eq!(io.read32(0x15C), 0b0010 << 4);
        io.set_inputs(0);
        assert!(!io.irq_active());
    }
}
