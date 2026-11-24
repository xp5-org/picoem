//! RP2040 / RP2350 USB controller in device mode, at transaction level.
//!
//! The controller is the same IP block on both chips: registers at
//! `USBCTRL_REGS_BASE`, 4 KB of dual-port RAM at `USBCTRL_DPRAM_BASE`
//! (datasheet RP2350 §12.7, RP2040 §4.1). Firmware sees the register and
//! DPRAM behaviour the TinyUSB `dcd_rp2040` driver relies on; the far end
//! of the cable is a host model, such as [`cdc_host::CdcHost`], that calls
//! [`UsbCtrl::bus_reset`], [`UsbCtrl::sof`], [`UsbCtrl::setup`],
//! [`UsbCtrl::in_token`] and [`UsbCtrl::out_token`] once per bus
//! transaction. Packets are whole: there is no bit-level timing, the host
//! model decides how far apart transactions are in emulated time.
//!
//! # DPRAM layout, device mode
//!
//! | Offset          | Contents                                          |
//! |-----------------|---------------------------------------------------|
//! | `0x000..0x008`  | last SETUP packet                                 |
//! | `0x008..0x080`  | endpoint control, EP1..EP15, IN then OUT, 8 bytes per endpoint |
//! | `0x080..0x100`  | buffer control, EP0..EP15, IN then OUT            |
//! | `0x100..0x140`  | EP0 buffer 0, shared by both directions           |
//! | `0x140..0x180`  | EP0 buffer 1                                      |
//! | `0x180..0x1000` | data buffers addressed by endpoint control        |
//!
//! A buffer control word holds two 16-bit halves, buffer 0 low and
//! buffer 1 high: `LEN[9:0]`, `AVAIL` bit 10, `STALL` bit 11, `RESET`
//! bit 12 which returns the buffer select to buffer 0, `PID` bit 13
//! for DATA1, `LAST` bit 14, `FULL` bit 15.
//!
//! # Transactions
//!
//! - SETUP: the 8 bytes land at DPRAM 0, `SIE_STATUS.SETUP_REC` is set
//!   and both EP0 stall-arm bits clear. The hardware always ACKs.
//! - IN: the selected buffer of the endpoint's IN half must be `AVAIL`
//!   and `FULL`. Its `LEN` bytes are sent with the buffer's PID, `AVAIL`
//!   and `FULL` clear and `LEN` stays. Otherwise NAK, or STALL when the
//!   `STALL` bit is set, which on EP0 also needs `EP_STALL_ARM`.
//! - OUT: the selected OUT buffer must be `AVAIL` and not `FULL`, and the
//!   data must fit `LEN`. The data is written, `LEN` becomes its length,
//!   `FULL` sets and `AVAIL` clears. A DATA PID that differs from the
//!   buffer's sets `DATA_SEQ_ERROR` and the packet is ACKed and dropped,
//!   which is how a device treats a retried packet it already has.
//! - A completed buffer sets its `BUFF_STATUS` bit, bit 2n for EPn IN and
//!   2n+1 for OUT, per buffer or per pair of buffers as the endpoint
//!   control, or `SIE_CTRL` for EP0, asks. `BUFF_CPU_SHOULD_HANDLE` holds
//!   which buffer finished. Double-buffered endpoints alternate buffers.
//!
//! Transactions address the device by `ADDR_ENDP.ADDRESS`; one sent to
//! another address, or to a controller that is not enabled with its D+
//! pull-up on, times out as on a real bus.
//!
//! # Interrupts
//!
//! `INTR` is derived: `BUFF_STATUS` non-zero, `SIE_STATUS` bits for
//! SETUP_REC, BUS_RESET, SUSPENDED and RESUME, `EP_ABORT_DONE` non-zero,
//! `EP_STATUS_STALL_NAK` non-zero, the SOF latch, which a `SOF_RD` read
//! clears, and the data sequence error. `INTS = (INTR | INTF) & INTE`
//! and the NVIC line is high while `INTS` is non-zero.
//!
//! `EP_ABORT` completes at once: nothing is in flight between
//! transactions, so `EP_ABORT_DONE` sets for every bit written.
//!
//! Host mode is not modelled.

pub mod cdc_host;

/// USB control-register base, both chips.
pub const USBCTRL_REGS_BASE: u32 = 0x5011_0000;
/// USB DPRAM base, both chips.
pub const USBCTRL_DPRAM_BASE: u32 = 0x5010_0000;
/// DPRAM size in bytes.
pub const USBCTRL_DPRAM_SIZE: u32 = 0x1000;
/// End of the setup packet, endpoint control and buffer control window.
pub const DPRAM_BUFFER_CONTROL_END: u32 = 0x180;

// --- Register offsets, pico-sdk hardware/regs/usb.h ---
pub const ADDR_ENDP: u32 = 0x000;
pub const MAIN_CTRL: u32 = 0x040;
pub const SOF_WR: u32 = 0x044;
pub const SOF_RD: u32 = 0x048;
pub const SIE_CTRL: u32 = 0x04C;
pub const SIE_STATUS: u32 = 0x050;
pub const INT_EP_CTRL: u32 = 0x054;
pub const BUFF_STATUS: u32 = 0x058;
pub const BUFF_CPU_SHOULD_HANDLE: u32 = 0x05C;
pub const EP_ABORT: u32 = 0x060;
pub const EP_ABORT_DONE: u32 = 0x064;
pub const EP_STALL_ARM: u32 = 0x068;
pub const NAK_POLL: u32 = 0x06C;
pub const EP_STATUS_STALL_NAK: u32 = 0x070;
pub const USB_MUXING: u32 = 0x074;
pub const USB_PWR: u32 = 0x078;
pub const USBPHY_DIRECT: u32 = 0x07C;
pub const USBPHY_DIRECT_OVERRIDE: u32 = 0x080;
pub const USBPHY_TRIM: u32 = 0x084;
pub const LINESTATE_TUNING: u32 = 0x088;
pub const INTR: u32 = 0x08C;
pub const INTE: u32 = 0x090;
pub const INTF: u32 = 0x094;
pub const INTS: u32 = 0x098;

// --- MAIN_CTRL ---
pub const MAIN_CTRL_CONTROLLER_EN: u32 = 1 << 0;
pub const MAIN_CTRL_HOST_NDEVICE: u32 = 1 << 1;

// --- SIE_CTRL ---
pub const SIE_CTRL_PULLUP_EN: u32 = 1 << 16;
pub const SIE_CTRL_EP0_INT_NAK: u32 = 1 << 27;
pub const SIE_CTRL_EP0_INT_2BUF: u32 = 1 << 28;
pub const SIE_CTRL_EP0_INT_1BUF: u32 = 1 << 29;
pub const SIE_CTRL_EP0_DOUBLE_BUF: u32 = 1 << 30;
pub const SIE_CTRL_EP0_INT_STALL: u32 = 1 << 31;

// --- SIE_STATUS ---
pub const SIE_STATUS_VBUS_DETECTED: u32 = 1 << 0;
pub const SIE_STATUS_SUSPENDED: u32 = 1 << 4;
pub const SIE_STATUS_RESUME: u32 = 1 << 11;
pub const SIE_STATUS_CONNECTED: u32 = 1 << 16;
pub const SIE_STATUS_SETUP_REC: u32 = 1 << 17;
pub const SIE_STATUS_TRANS_COMPLETE: u32 = 1 << 18;
pub const SIE_STATUS_BUS_RESET: u32 = 1 << 19;
pub const SIE_STATUS_DATA_SEQ_ERROR: u32 = 1 << 31;
/// SIE_STATUS bits a write of 1 clears: the error bits 23..31,
/// BUS_RESET, TRANS_COMPLETE, SETUP_REC, CONNECTED, RX_SHORT_PACKET,
/// RESUME, VBUS_OVER_CURR, SUSPENDED.
const SIE_STATUS_WC: u32 = 0xFF80_0000 | (1 << 19) | (1 << 18) | (1 << 17) | (1 << 16) | (1 << 12) | (1 << 11) | (1 << 10) | (1 << 4);

// --- USB_PWR ---
pub const USB_PWR_VBUS_DETECT: u32 = 1 << 2;
pub const USB_PWR_VBUS_DETECT_OVERRIDE_EN: u32 = 1 << 3;

// --- INTR / INTE / INTF / INTS ---
pub const INT_ERROR_DATA_SEQ: u32 = 1 << 5;
pub const INT_BUFF_STATUS: u32 = 1 << 4;
pub const INT_BUS_RESET: u32 = 1 << 12;
pub const INT_DEV_CONN_DIS: u32 = 1 << 13;
pub const INT_DEV_SUSPEND: u32 = 1 << 14;
pub const INT_DEV_RESUME_FROM_HOST: u32 = 1 << 15;
pub const INT_SETUP_REQ: u32 = 1 << 16;
pub const INT_DEV_SOF: u32 = 1 << 17;
pub const INT_ABORT_DONE: u32 = 1 << 18;
pub const INT_EP_STALL_NAK: u32 = 1 << 19;

// --- DPRAM ---
pub const DPRAM_SETUP: usize = 0x000;
pub const DPRAM_EP_CTRL: usize = 0x008;
pub const DPRAM_BUF_CTRL: usize = 0x080;
pub const DPRAM_EP0_BUF0: usize = 0x100;
pub const DPRAM_EP0_BUF1: usize = 0x140;

// --- buffer control, one 16-bit half ---
pub const BUF_LEN_MASK: u32 = 0x3FF;
pub const BUF_AVAIL: u32 = 1 << 10;
pub const BUF_STALL: u32 = 1 << 11;
pub const BUF_RESET_SEL: u32 = 1 << 12;
pub const BUF_DATA1_PID: u32 = 1 << 13;
pub const BUF_LAST: u32 = 1 << 14;
pub const BUF_FULL: u32 = 1 << 15;

// --- endpoint control ---
pub const EP_CTRL_ENABLE: u32 = 1 << 31;
pub const EP_CTRL_DOUBLE_BUFFERED: u32 = 1 << 30;
pub const EP_CTRL_INT_PER_BUFFER: u32 = 1 << 29;
pub const EP_CTRL_INT_PER_DOUBLE_BUFFER: u32 = 1 << 28;
pub const EP_CTRL_INT_ON_STALL: u32 = 1 << 17;
pub const EP_CTRL_INT_ON_NAK: u32 = 1 << 16;
pub const EP_CTRL_BUFFER_ADDRESS: u32 = 0xFFC0;

/// Direction of a data transaction, from the host's point of view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    In,
    Out,
}

/// What the device answered to one host transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UsbResp {
    /// SETUP or OUT accepted.
    Ack,
    /// IN data and its DATA PID, 0 or 1.
    Data(Vec<u8>, u8),
    Nak,
    Stall,
    /// No answer: wrong address, endpoint disabled, controller off.
    Timeout,
}

/// Counters a test can read back.
#[derive(Clone, Debug, Default)]
pub struct UsbStats {
    pub setups: u64,
    pub in_data: u64,
    pub out_acks: u64,
    pub naks: u64,
    pub stalls: u64,
    pub timeouts: u64,
    pub seq_errors: u64,
    pub sofs: u64,
    pub bus_resets: u64,
}

/// USB controller, device mode. See module docs.
pub struct UsbCtrl {
    addr_endp: u32,
    addr_endp_n: [u32; 15],
    main_ctrl: u32,
    main_ctrl_reset: u32,
    sof_wr: u32,
    sof_rd: u32,
    sie_ctrl: u32,
    sie_ctrl_reset: u32,
    sie_status: u32,
    int_ep_ctrl: u32,
    buff_status: u32,
    buff_cpu_should_handle: u32,
    ep_abort: u32,
    ep_abort_done: u32,
    ep_stall_arm: u32,
    nak_poll: u32,
    ep_status_stall_nak: u32,
    usb_muxing: u32,
    usb_pwr: u32,
    usbphy_direct: u32,
    usbphy_direct_override: u32,
    usbphy_trim: u32,
    linestate_tuning: u32,
    inte: u32,
    intf: u32,
    /// SOF received since the last SOF_RD read.
    sof_latch: bool,
    /// Connection changed since firmware last cleared CONNECTED.
    conn_latch: bool,
    /// A host is on the far end of the cable, VBUS present.
    host_present: bool,
    /// Next buffer per endpoint direction, index 2n for IN, 2n+1 for OUT.
    next_sel: [u8; 32],
    /// Buffers completed since the last interrupt, for per-pair interrupts.
    done_count: [u8; 32],
    dpram: Box<[u8; 4096]>,
    pub stats: UsbStats,
}

fn rmw(stored: &mut u32, value: u32, alias: u32) {
    match alias {
        1 => *stored ^= value,
        2 => *stored |= value,
        3 => *stored &= !value,
        _ => *stored = value,
    }
}

impl UsbCtrl {
    /// RP2350 reset values: MAIN_CTRL.PHY_ISO and SIE_CTRL.PULLDOWN_EN set.
    pub fn new() -> Self {
        Self::with_reset_values(0x0000_0004, 0x0000_8000)
    }

    /// RP2040 reset values: MAIN_CTRL and SIE_CTRL read 0.
    pub fn new_rp2040() -> Self {
        Self::with_reset_values(0, 0)
    }

    fn with_reset_values(main_ctrl_reset: u32, sie_ctrl_reset: u32) -> Self {
        Self {
            addr_endp: 0,
            addr_endp_n: [0; 15],
            main_ctrl: main_ctrl_reset,
            main_ctrl_reset,
            sof_wr: 0,
            sof_rd: 0,
            sie_ctrl: sie_ctrl_reset,
            sie_ctrl_reset,
            sie_status: 0,
            int_ep_ctrl: 0,
            buff_status: 0,
            buff_cpu_should_handle: 0,
            ep_abort: 0,
            ep_abort_done: 0,
            ep_stall_arm: 0,
            nak_poll: 0,
            ep_status_stall_nak: 0,
            usb_muxing: 0,
            usb_pwr: 0,
            usbphy_direct: 0,
            usbphy_direct_override: 0,
            usbphy_trim: 0,
            linestate_tuning: 0,
            inte: 0,
            intf: 0,
            sof_latch: false,
            conn_latch: false,
            host_present: false,
            next_sel: [0; 32],
            done_count: [0; 32],
            dpram: Box::new([0u8; 4096]),
            stats: UsbStats::default(),
        }
    }

    /// Block reset through RESETS: every register returns to its reset
    /// value and the setup, endpoint control and buffer control window
    /// clears. The data buffers keep their contents. A host stays
    /// plugged in.
    pub fn reset(&mut self) {
        let host = self.host_present;
        let stats = std::mem::take(&mut self.stats);
        let dpram = std::mem::replace(&mut self.dpram, Box::new([0u8; 4096]));
        *self = Self::with_reset_values(self.main_ctrl_reset, self.sie_ctrl_reset);
        self.dpram = dpram;
        for b in &mut self.dpram[..DPRAM_BUFFER_CONTROL_END as usize] {
            *b = 0;
        }
        self.host_present = host;
        self.stats = stats;
    }

    // ------------------------------------------------------------ registers

    fn sie_status_read(&self) -> u32 {
        let mut v = self.sie_status;
        let vbus = if self.usb_pwr & USB_PWR_VBUS_DETECT_OVERRIDE_EN != 0 {
            self.usb_pwr & USB_PWR_VBUS_DETECT != 0
        } else {
            self.host_present
        };
        if vbus {
            v |= SIE_STATUS_VBUS_DETECTED;
        }
        v
    }

    /// `INTR`, derived from the status registers.
    pub fn intr(&self) -> u32 {
        let mut v = 0;
        if self.buff_status != 0 {
            v |= INT_BUFF_STATUS;
        }
        if self.sie_status & SIE_STATUS_DATA_SEQ_ERROR != 0 {
            v |= INT_ERROR_DATA_SEQ;
        }
        if self.sie_status & SIE_STATUS_BUS_RESET != 0 {
            v |= INT_BUS_RESET;
        }
        if self.conn_latch {
            v |= INT_DEV_CONN_DIS;
        }
        if self.sie_status & SIE_STATUS_SUSPENDED != 0 {
            v |= INT_DEV_SUSPEND;
        }
        if self.sie_status & SIE_STATUS_RESUME != 0 {
            v |= INT_DEV_RESUME_FROM_HOST;
        }
        if self.sie_status & SIE_STATUS_SETUP_REC != 0 {
            v |= INT_SETUP_REQ;
        }
        if self.sof_latch {
            v |= INT_DEV_SOF;
        }
        if self.ep_abort_done != 0 {
            v |= INT_ABORT_DONE;
        }
        if self.ep_status_stall_nak != 0 {
            v |= INT_EP_STALL_NAK;
        }
        v
    }

    /// `INTS`: what reaches the NVIC.
    pub fn ints(&self) -> u32 {
        (self.intr() | self.intf) & self.inte
    }

    /// Level of the USBCTRL NVIC line.
    pub fn irq_level(&self) -> bool {
        self.ints() != 0
    }

    /// Read a register by offset within the 4 KB register window.
    pub fn read32(&mut self, offset: u32) -> u32 {
        match offset {
            ADDR_ENDP => self.addr_endp,
            0x004..=0x03C => self.addr_endp_n[(offset as usize - 4) / 4],
            MAIN_CTRL => self.main_ctrl,
            SOF_WR => self.sof_wr,
            SOF_RD => {
                self.sof_latch = false;
                self.sof_rd
            }
            SIE_CTRL => self.sie_ctrl,
            SIE_STATUS => self.sie_status_read(),
            INT_EP_CTRL => self.int_ep_ctrl,
            BUFF_STATUS => self.buff_status,
            BUFF_CPU_SHOULD_HANDLE => self.buff_cpu_should_handle,
            EP_ABORT => self.ep_abort,
            EP_ABORT_DONE => self.ep_abort_done,
            EP_STALL_ARM => self.ep_stall_arm,
            NAK_POLL => self.nak_poll,
            EP_STATUS_STALL_NAK => self.ep_status_stall_nak,
            USB_MUXING => self.usb_muxing,
            USB_PWR => self.usb_pwr,
            USBPHY_DIRECT => self.usbphy_direct,
            USBPHY_DIRECT_OVERRIDE => self.usbphy_direct_override,
            USBPHY_TRIM => self.usbphy_trim,
            LINESTATE_TUNING => self.linestate_tuning,
            INTR => self.intr(),
            INTE => self.inte,
            INTF => self.intf,
            INTS => self.ints(),
            _ => 0,
        }
    }

    /// Write a register. `alias` is the APB atomic alias: 0 normal,
    /// 1 XOR, 2 SET, 3 CLR. On the write-1-to-clear registers every
    /// alias clears the bits written, which is how the SDK's
    /// `usb_hw_clear->buf_status = bit` works.
    pub fn write32(&mut self, offset: u32, value: u32, alias: u32) {
        match offset {
            ADDR_ENDP => rmw(&mut self.addr_endp, value, alias),
            0x004..=0x03C => rmw(&mut self.addr_endp_n[(offset as usize - 4) / 4], value, alias),
            MAIN_CTRL => rmw(&mut self.main_ctrl, value, alias),
            SOF_WR => rmw(&mut self.sof_wr, value, alias),
            SOF_RD => {}
            SIE_CTRL => rmw(&mut self.sie_ctrl, value, alias),
            SIE_STATUS => {
                let clr = value & SIE_STATUS_WC;
                if clr & SIE_STATUS_CONNECTED != 0 {
                    self.conn_latch = false;
                }
                // CONNECTED is a status bit as well as the connect-change
                // clear: it stays while the device is on the bus.
                self.sie_status &= !(clr & !SIE_STATUS_CONNECTED);
            }
            INT_EP_CTRL => rmw(&mut self.int_ep_ctrl, value, alias),
            BUFF_STATUS => self.buff_status &= !value,
            BUFF_CPU_SHOULD_HANDLE => {}
            EP_ABORT => {
                rmw(&mut self.ep_abort, value, alias);
                // Nothing is in flight between transactions.
                self.ep_abort_done |= self.ep_abort;
            }
            EP_ABORT_DONE => self.ep_abort_done &= !value,
            EP_STALL_ARM => rmw(&mut self.ep_stall_arm, value, alias),
            NAK_POLL => rmw(&mut self.nak_poll, value, alias),
            EP_STATUS_STALL_NAK => self.ep_status_stall_nak &= !value,
            USB_MUXING => rmw(&mut self.usb_muxing, value, alias),
            USB_PWR => rmw(&mut self.usb_pwr, value, alias),
            USBPHY_DIRECT => rmw(&mut self.usbphy_direct, value, alias),
            USBPHY_DIRECT_OVERRIDE => rmw(&mut self.usbphy_direct_override, value, alias),
            USBPHY_TRIM => rmw(&mut self.usbphy_trim, value, alias),
            LINESTATE_TUNING => rmw(&mut self.linestate_tuning, value, alias),
            INTR | INTS => {}
            INTE => rmw(&mut self.inte, value, alias),
            INTF => rmw(&mut self.intf, value, alias),
            _ => {}
        }
    }

    // ---------------------------------------------------------------- DPRAM

    /// Read 1, 2 or 4 bytes of DPRAM at `offset`, zero-extended.
    pub fn read_dpram(&self, offset: u32, size: u8) -> u32 {
        let off = (offset & (USBCTRL_DPRAM_SIZE - 1)) as usize;
        let n = match size {
            1 | 2 | 4 => size as usize,
            _ => return 0,
        };
        let mut v = 0u32;
        for i in 0..n {
            v |= (self.dpram[(off + i) & 0xFFF] as u32) << (8 * i);
        }
        v
    }

    /// Write 1, 2 or 4 bytes of DPRAM at `offset`. A write to a buffer
    /// control word with `RESET` set in its low half returns that
    /// endpoint's buffer select to buffer 0.
    pub fn write_dpram(&mut self, offset: u32, value: u32, size: u8) {
        let off = (offset & (USBCTRL_DPRAM_SIZE - 1)) as usize;
        let n = match size {
            1 | 2 | 4 => size as usize,
            _ => return,
        };
        for i in 0..n {
            self.dpram[(off + i) & 0xFFF] = (value >> (8 * i)) as u8;
        }
        if (DPRAM_BUF_CTRL..DPRAM_EP0_BUF0).contains(&off) {
            let word = off & !3;
            let idx = (word - DPRAM_BUF_CTRL) / 4; // 2n IN, 2n+1 OUT
            if self.word(word) & BUF_RESET_SEL != 0 {
                self.next_sel[idx] = 0;
                self.done_count[idx] = 0;
            }
        }
    }

    fn word(&self, off: usize) -> u32 {
        u32::from_le_bytes(self.dpram[off..off + 4].try_into().unwrap())
    }

    fn set_word(&mut self, off: usize, v: u32) {
        self.dpram[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    // ---------------------------------------------------------- host side

    /// Plug a host in, or pull it out: VBUS follows. Pulling the host
    /// out clears CONNECTED.
    pub fn set_host_present(&mut self, present: bool) {
        if self.host_present != present && !present && self.sie_status & SIE_STATUS_CONNECTED != 0 {
            self.sie_status &= !SIE_STATUS_CONNECTED;
            self.conn_latch = true;
        }
        self.host_present = present;
    }

    /// The device is enabled in device mode with its D+ pull-up on, so a
    /// host sees it attached.
    pub fn device_attached(&self) -> bool {
        self.host_present
            && self.main_ctrl & MAIN_CTRL_CONTROLLER_EN != 0
            && self.main_ctrl & MAIN_CTRL_HOST_NDEVICE == 0
            && self.sie_ctrl & SIE_CTRL_PULLUP_EN != 0
    }

    /// Device address the firmware programmed.
    pub fn dev_addr(&self) -> u8 {
        (self.addr_endp & 0x7F) as u8
    }

    /// The host drove a bus reset: BUS_RESET sets and the device is
    /// connected at full speed.
    pub fn bus_reset(&mut self) {
        if !self.device_attached() {
            return;
        }
        self.stats.bus_resets += 1;
        self.sie_status |= SIE_STATUS_BUS_RESET;
        if self.sie_status & SIE_STATUS_CONNECTED == 0 {
            self.sie_status |= SIE_STATUS_CONNECTED;
            self.conn_latch = true;
        }
        self.sie_status &= !SIE_STATUS_SUSPENDED;
    }

    /// Start of frame `frame` (11 bits).
    pub fn sof(&mut self, frame: u16) {
        if !self.device_attached() {
            return;
        }
        self.stats.sofs += 1;
        self.sof_rd = (frame & 0x7FF) as u32;
        self.sof_latch = true;
    }

    fn addressed(&mut self, addr: u8) -> bool {
        if self.device_attached() && addr == self.dev_addr() {
            true
        } else {
            self.stats.timeouts += 1;
            false
        }
    }

    /// SETUP packet to EP0 of device `addr`.
    pub fn setup(&mut self, addr: u8, pkt: &[u8; 8]) -> UsbResp {
        if !self.addressed(addr) {
            return UsbResp::Timeout;
        }
        self.stats.setups += 1;
        self.dpram[DPRAM_SETUP..DPRAM_SETUP + 8].copy_from_slice(pkt);
        self.sie_status |= SIE_STATUS_SETUP_REC;
        self.ep_stall_arm &= !0x3;
        UsbResp::Ack
    }

    /// Endpoint control word for `ep` and `dir`, None for EP0.
    fn ep_ctrl(&self, ep: u8, dir: Dir) -> Option<u32> {
        if ep == 0 {
            return None;
        }
        let off = DPRAM_EP_CTRL + (ep as usize - 1) * 8 + if dir == Dir::In { 0 } else { 4 };
        Some(self.word(off))
    }

    fn buf_ctrl_off(ep: u8, dir: Dir) -> usize {
        DPRAM_BUF_CTRL + ep as usize * 8 + if dir == Dir::In { 0 } else { 4 }
    }

    /// DPRAM offset of buffer `sel` of `ep`.
    fn buffer_off(&self, ep: u8, dir: Dir, sel: u8) -> usize {
        match self.ep_ctrl(ep, dir) {
            None => if sel == 0 { DPRAM_EP0_BUF0 } else { DPRAM_EP0_BUF1 },
            Some(c) => (c & EP_CTRL_BUFFER_ADDRESS) as usize + 64 * sel as usize,
        }
    }

    fn double_buffered(&self, ep: u8, dir: Dir) -> bool {
        match self.ep_ctrl(ep, dir) {
            None => self.sie_ctrl & SIE_CTRL_EP0_DOUBLE_BUF != 0,
            Some(c) => c & EP_CTRL_DOUBLE_BUFFERED != 0,
        }
    }

    /// The endpoint answers at all: EP0 always, others when enabled.
    fn ep_enabled(&self, ep: u8, dir: Dir) -> bool {
        match self.ep_ctrl(ep, dir) {
            None => ep == 0,
            Some(c) => c & EP_CTRL_ENABLE != 0,
        }
    }

    fn stalled(&self, ep: u8, dir: Dir, half: u32) -> bool {
        if half & BUF_STALL == 0 {
            return false;
        }
        if ep == 0 {
            let arm = if dir == Dir::In { 1 } else { 2 };
            return self.ep_stall_arm & arm != 0;
        }
        true
    }

    fn note_stall_nak(&mut self, ep: u8, dir: Dir, stall: bool) {
        let want = match self.ep_ctrl(ep, dir) {
            None => self.sie_ctrl & if stall { SIE_CTRL_EP0_INT_STALL } else { SIE_CTRL_EP0_INT_NAK } != 0,
            Some(c) => c & if stall { EP_CTRL_INT_ON_STALL } else { EP_CTRL_INT_ON_NAK } != 0,
        };
        if want {
            let bit = 2 * ep as u32 + if dir == Dir::Out { 1 } else { 0 };
            self.ep_status_stall_nak |= 1 << bit;
        }
    }

    /// A buffer of `ep` finished: advance the select and raise BUFF_STATUS
    /// as the endpoint's interrupt setting asks.
    fn buffer_done(&mut self, ep: u8, dir: Dir, sel: u8) {
        let idx = 2 * ep as usize + if dir == Dir::Out { 1 } else { 0 };
        let double = self.double_buffered(ep, dir);
        self.next_sel[idx] = if double { sel ^ 1 } else { 0 };
        self.done_count[idx] = self.done_count[idx].wrapping_add(1);
        let (per_buf, per_pair) = match self.ep_ctrl(ep, dir) {
            None => (self.sie_ctrl & SIE_CTRL_EP0_INT_1BUF != 0, self.sie_ctrl & SIE_CTRL_EP0_INT_2BUF != 0),
            Some(c) => (c & EP_CTRL_INT_PER_BUFFER != 0, c & EP_CTRL_INT_PER_DOUBLE_BUFFER != 0),
        };
        let fire = per_buf || (per_pair && double && self.done_count[idx] % 2 == 0);
        if fire {
            self.done_count[idx] = 0;
            self.buff_status |= 1 << idx;
            if sel == 1 {
                self.buff_cpu_should_handle |= 1 << idx;
            } else {
                self.buff_cpu_should_handle &= !(1 << idx);
            }
        }
    }

    /// IN token to `ep` of device `addr`.
    pub fn in_token(&mut self, addr: u8, ep: u8) -> UsbResp {
        if !self.addressed(addr) {
            return UsbResp::Timeout;
        }
        if ep > 15 || !self.ep_enabled(ep, Dir::In) {
            self.stats.timeouts += 1;
            return UsbResp::Timeout;
        }
        let bc_off = Self::buf_ctrl_off(ep, Dir::In);
        let bc = self.word(bc_off);
        let sel = self.next_sel[2 * ep as usize];
        let shift = 16 * sel as u32;
        let half = (bc >> shift) & 0xFFFF;
        if self.stalled(ep, Dir::In, bc & 0xFFFF) {
            self.stats.stalls += 1;
            self.note_stall_nak(ep, Dir::In, true);
            return UsbResp::Stall;
        }
        if half & BUF_AVAIL == 0 || half & BUF_FULL == 0 {
            self.stats.naks += 1;
            self.note_stall_nak(ep, Dir::In, false);
            return UsbResp::Nak;
        }
        let len = (half & BUF_LEN_MASK) as usize;
        let off = self.buffer_off(ep, Dir::In, sel);
        let data: Vec<u8> = (0..len).map(|i| self.dpram[(off + i) & 0xFFF]).collect();
        let pid = ((half & BUF_DATA1_PID) != 0) as u8;
        let new_half = half & !(BUF_AVAIL | BUF_FULL);
        self.set_word(bc_off, (bc & !(0xFFFF << shift)) | (new_half << shift));
        self.stats.in_data += 1;
        self.buffer_done(ep, Dir::In, sel);
        UsbResp::Data(data, pid)
    }

    /// OUT token and DATA `pid` packet to `ep` of device `addr`.
    pub fn out_token(&mut self, addr: u8, ep: u8, data: &[u8], pid: u8) -> UsbResp {
        if !self.addressed(addr) {
            return UsbResp::Timeout;
        }
        if ep > 15 || !self.ep_enabled(ep, Dir::Out) || data.len() > 64 {
            self.stats.timeouts += 1;
            return UsbResp::Timeout;
        }
        let bc_off = Self::buf_ctrl_off(ep, Dir::Out);
        let bc = self.word(bc_off);
        let sel = self.next_sel[2 * ep as usize + 1];
        let shift = 16 * sel as u32;
        let half = (bc >> shift) & 0xFFFF;
        if self.stalled(ep, Dir::Out, bc & 0xFFFF) {
            self.stats.stalls += 1;
            self.note_stall_nak(ep, Dir::Out, true);
            return UsbResp::Stall;
        }
        if half & BUF_AVAIL == 0 || half & BUF_FULL != 0 || data.len() > (half & BUF_LEN_MASK) as usize {
            self.stats.naks += 1;
            self.note_stall_nak(ep, Dir::Out, false);
            return UsbResp::Nak;
        }
        let want_pid = ((half & BUF_DATA1_PID) != 0) as u8;
        if pid != want_pid {
            self.stats.seq_errors += 1;
            self.sie_status |= SIE_STATUS_DATA_SEQ_ERROR;
            return UsbResp::Ack;
        }
        let off = self.buffer_off(ep, Dir::Out, sel);
        for (i, b) in data.iter().enumerate() {
            self.dpram[(off + i) & 0xFFF] = *b;
        }
        let new_half = (half & !(BUF_AVAIL | BUF_LEN_MASK)) | BUF_FULL | data.len() as u32;
        self.set_word(bc_off, (bc & !(0xFFFF << shift)) | (new_half << shift));
        self.stats.out_acks += 1;
        self.buffer_done(ep, Dir::Out, sel);
        UsbResp::Ack
    }
}

impl Default for UsbCtrl {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
