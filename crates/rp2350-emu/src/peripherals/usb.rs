//! RP2350 USB controller, device mode.
//!
//! The controller model is [`picoem_common::usb::UsbCtrl`], shared with
//! the RP2040: registers at `USBCTRL_REGS_BASE`, DPRAM at
//! `USBCTRL_DPRAM_BASE`, and a transaction-level host side that a host
//! model such as [`picoem_common::usb::cdc_host::CdcHost`] drives. This
//! wrapper adds the RP2350 wiring: the NVIC line is
//! [`crate::irq::IRQ_USBCTRL_IRQ`], and register writes report that line
//! in the bus's `u64` IRQ mask.
//!
//! The bus asserts the line from `tick_peripherals` while `INTS` is
//! non-zero and after each [`crate::Bus::usb_host_poll`], and resets the
//! controller when firmware asserts `RESETS.USBCTRL`. The threaded
//! execution model carries the registers and DPRAM but never raises the
//! interrupt, so USB device firmware runs in the Serial model.

use std::ops::{Deref, DerefMut};

pub use picoem_common::usb::*;

use crate::irq::IRQ_USBCTRL_IRQ;

/// RP2350 USB controller: the shared model plus the RP2350 IRQ line.
pub struct UsbCtrl(pub picoem_common::usb::UsbCtrl);

impl UsbCtrl {
    /// Post-reset state with RP2350 reset values.
    pub fn new() -> Self {
        Self(picoem_common::usb::UsbCtrl::new())
    }

    /// Write a register; `irqs` gets the USBCTRL bit when the line is high
    /// afterwards.
    pub fn write32(&mut self, offset: u32, value: u32, alias: u32, irqs: &mut u64) {
        self.0.write32(offset, value, alias);
        if self.0.irq_level() {
            *irqs |= 1u64 << IRQ_USBCTRL_IRQ;
        }
    }
}

impl Default for UsbCtrl {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for UsbCtrl {
    type Target = picoem_common::usb::UsbCtrl;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for UsbCtrl {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::RESET_USBCTRL;
    use crate::{Bus, Config, Emulator};
    use picoem_common::usb::cdc_host::{CdcHost, HostState};

    const RESETS_BASE: u32 = 0x4002_0000;
    const RESETS_SET: u32 = RESETS_BASE + 0x2000;
    const RESETS_CLR: u32 = RESETS_BASE + 0x3000;

    fn release_usbctrl(bus: &mut Bus) {
        bus.write32(RESETS_CLR, 1 << RESET_USBCTRL, 0);
    }

    #[test]
    fn bus_routes_main_ctrl_round_trip() {
        let mut bus = Bus::new();
        release_usbctrl(&mut bus);
        bus.write32(USBCTRL_REGS_BASE + MAIN_CTRL, 0x0000_0001, 0);
        assert_eq!(bus.read32(USBCTRL_REGS_BASE + MAIN_CTRL, 0), 0x0000_0001);
    }

    #[test]
    fn bus_dpram_byte_halfword_round_trip() {
        let mut bus = Bus::new();
        release_usbctrl(&mut bus);
        bus.write8(USBCTRL_DPRAM_BASE + 0x300, 0x11, 0);
        bus.write8(USBCTRL_DPRAM_BASE + 0x301, 0x22, 0);
        bus.write16(USBCTRL_DPRAM_BASE + 0x302, 0x4433, 0);
        assert_eq!(bus.read32(USBCTRL_DPRAM_BASE + 0x300, 0), 0x4433_2211);
        assert_eq!(bus.read16(USBCTRL_DPRAM_BASE + 0x300, 0), 0x2211);
        assert_eq!(bus.read8(USBCTRL_DPRAM_BASE + 0x303, 0), 0x44);
    }

    #[test]
    fn bus_reads_zero_while_held_in_reset() {
        let mut bus = Bus::new();
        bus.write32(USBCTRL_REGS_BASE + MAIN_CTRL, 0x1, 0);
        assert_eq!(bus.read32(USBCTRL_REGS_BASE + MAIN_CTRL, 0), 0);
    }

    #[test]
    fn resets_assert_clears_the_controller() {
        let mut bus = Bus::new();
        release_usbctrl(&mut bus);
        bus.write32(USBCTRL_REGS_BASE + INTE, INT_BUS_RESET, 0);
        bus.write32(USBCTRL_DPRAM_BASE + 0x80, 0xFFFF, 0);
        bus.write32(RESETS_SET, 1 << RESET_USBCTRL, 0);
        release_usbctrl(&mut bus);
        assert_eq!(bus.read32(USBCTRL_REGS_BASE + INTE, 0), 0);
        assert_eq!(bus.read32(USBCTRL_DPRAM_BASE + 0x80, 0), 0);
        assert_eq!(bus.read32(USBCTRL_REGS_BASE + MAIN_CTRL, 0), 0x4);
    }

    /// SIE_STATUS bits clear through the clear alias, the way the SDK's
    /// `usb_hw_clear->sie_status = bit` writes them.
    #[test]
    fn sie_status_clears_through_clear_alias() {
        let mut emu = Emulator::new(Config::default());
        release_usbctrl(&mut emu.bus);
        emu.bus.write32(USBCTRL_REGS_BASE + MAIN_CTRL, MAIN_CTRL_CONTROLLER_EN, 0);
        emu.bus.write32(USBCTRL_REGS_BASE + SIE_CTRL, SIE_CTRL_PULLUP_EN, 0);
        emu.bus.usb_plug(true);
        emu.bus.usb().bus_reset();
        assert_ne!(emu.bus.read32(USBCTRL_REGS_BASE + SIE_STATUS, 0) & SIE_STATUS_BUS_RESET, 0);
        emu.bus.write32(USBCTRL_REGS_BASE + 0x3000 + SIE_STATUS, SIE_STATUS_BUS_RESET, 0);
        assert_eq!(emu.bus.read32(USBCTRL_REGS_BASE + SIE_STATUS, 0) & SIE_STATUS_BUS_RESET, 0);
    }

    /// The NVIC line follows INTS: low through init, pending once a bus
    /// reset arrives with BUS_RESET enabled.
    #[test]
    fn usb_irq_follows_ints() {
        let mut emu = Emulator::new(Config::default());
        release_usbctrl(&mut emu.bus);
        emu.bus.write32(USBCTRL_REGS_BASE + USB_MUXING, 0x9, 0);
        emu.bus.write32(USBCTRL_REGS_BASE + USB_PWR, USB_PWR_VBUS_DETECT | USB_PWR_VBUS_DETECT_OVERRIDE_EN, 0);
        emu.bus.write32(USBCTRL_REGS_BASE + MAIN_CTRL, MAIN_CTRL_CONTROLLER_EN, 0);
        emu.bus.write32(USBCTRL_REGS_BASE + SIE_CTRL, SIE_CTRL_PULLUP_EN, 0);
        emu.bus.write32(USBCTRL_REGS_BASE + INTE, INT_BUS_RESET | INT_SETUP_REQ | INT_BUFF_STATUS, 0);
        for core in 0..2usize {
            assert_eq!(emu.bus.atomics.irq_pending_load(core) & (1u64 << IRQ_USBCTRL_IRQ), 0);
        }
        let mut host = CdcHost::new();
        let mut t = 0;
        while host.state != HostState::Enumerating && t < 200_000 {
            emu.bus.usb_host_poll(&mut host, t);
            t += 100;
        }
        assert_eq!(host.state, HostState::Enumerating);
        let pending = emu.bus.atomics.irq_pending_load(0);
        assert_ne!(pending & (1u64 << IRQ_USBCTRL_IRQ), 0, "bus reset pends USBCTRL_IRQ");
    }
}
