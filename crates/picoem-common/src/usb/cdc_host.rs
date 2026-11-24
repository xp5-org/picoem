//! A full-speed USB host with one CDC-ACM device on its port, driving a
//! [`UsbCtrl`] the way a PC's host controller and the Linux `cdc_acm`
//! driver would.
//!
//! The host is clocked by emulated time. The caller passes the current
//! time in microseconds to [`CdcHost::poll`] as often as it likes, once
//! per emulator quantum for example; the host sends a SOF at the start
//! of every 1 ms frame and at most one transaction per
//! [`CdcHost::slot_us`] for that transaction's size, so a bulk packet of
//! 64 bytes occupies about 57 us of the bus, as at 12 Mbit/s with
//! protocol overhead.
//!
//! Sequence after the device's pull-up appears:
//! 1. wait 100 ms for the connect debounce, then hold a 10 ms bus reset;
//! 2. GET_DESCRIPTOR device, 64 bytes, at address 0;
//! 3. SET_ADDRESS, then 2 ms recovery;
//! 4. GET_DESCRIPTOR device, configuration header, whole configuration;
//! 5. string descriptor 0, then the manufacturer, product and serial
//!    strings in language 0x0409;
//! 6. SET_CONFIGURATION;
//! 7. CDC SET_LINE_CODING with [`CdcHost::line_coding`], then
//!    SET_CONTROL_LINE_STATE with DTR and RTS, which is what a terminal
//!    program opening the tty does.
//!
//! After that the host keeps bulk IN polled, as `cdc_acm` keeps its read
//! URBs queued, and sends what [`CdcHost::write`] queued as bulk OUT
//! packets of the endpoint's max packet size. Data toggles are checked on
//! every IN packet; a wrong toggle is counted as a sequence error and the
//! packet dropped, as a host controller does with a repeat.
//!
//! A control transfer that has not finished after 5 s of emulated time
//! fails the enumeration with a message in [`CdcHost::error`].

use std::collections::VecDeque;

use super::{UsbCtrl, UsbResp};

/// Where the host is in bringing the device up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostState {
    /// No pull-up seen.
    Detached,
    /// Pull-up seen at this time, debouncing.
    Debounce(u64),
    /// Bus reset held until this time.
    Reset(u64),
    /// Running the enumeration script.
    Enumerating,
    /// Configured with DTR set: bulk data flows.
    Ready,
    /// Enumeration failed, see `error`.
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Setup,
    DataIn,
    DataOut,
    StatusIn,
    StatusOut,
}

struct Ctrl {
    addr: u8,
    setup: [u8; 8],
    out: Vec<u8>,
    out_pos: usize,
    in_len: usize,
    got: Vec<u8>,
    stage: Stage,
    toggle: u8,
    started_us: u64,
}

/// Steps of the enumeration script.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    DevDesc8,
    SetAddress,
    AddressWait(u64),
    DevDesc,
    ConfHeader,
    ConfAll,
    LangIds,
    StrManufacturer,
    StrProduct,
    StrSerial,
    SetConfig,
    LineCoding,
    LineState,
    Done,
}

/// Device facts read during enumeration.
#[derive(Clone, Debug, Default)]
pub struct DeviceInfo {
    pub vid: u16,
    pub pid: u16,
    pub bcd_device: u16,
    pub max_packet0: u8,
    pub manufacturer: String,
    pub product: String,
    pub serial: String,
    pub config_value: u8,
    pub comm_iface: u8,
    pub data_iface: u8,
    pub ep_in: u8,
    pub ep_out: u8,
    pub ep_in_size: u16,
    pub ep_out_size: u16,
    pub i_manufacturer: u8,
    pub i_product: u8,
    pub i_serial: u8,
}

/// Bus traffic counters, host side.
#[derive(Clone, Debug, Default)]
pub struct HostStats {
    pub control_transfers: u64,
    pub bulk_out_packets: u64,
    pub bulk_out_bytes: u64,
    pub bulk_in_packets: u64,
    pub bulk_in_bytes: u64,
    pub naks: u64,
    pub seq_errors: u64,
    pub stalls: u64,
    pub timeouts: u64,
}

/// USB host with a CDC-ACM device. See module docs.
pub struct CdcHost {
    pub state: HostState,
    /// Address given to the device.
    pub address: u8,
    /// SET_LINE_CODING payload: 115200 baud, 1 stop bit, no parity, 8 bits.
    /// A baud rate of 1200 would ask a Pico SDK device to reboot into
    /// BOOTSEL, so it is not the default.
    pub line_coding: [u8; 7],
    pub info: DeviceInfo,
    pub stats: HostStats,
    /// Enumeration log: one line per completed step.
    pub log: Vec<String>,
    pub error: Option<String>,
    /// When the device reached `Ready`, in us.
    pub ready_at_us: Option<u64>,
    frame: u16,
    next_frame_us: u64,
    next_txn_us: u64,
    step: Step,
    ctrl: Option<Ctrl>,
    tx: VecDeque<u8>,
    rx: Vec<u8>,
    in_toggle: u8,
    out_toggle: u8,
    /// Alternates bulk OUT and bulk IN when both have work.
    out_turn: bool,
    /// wTotalLength of the configuration descriptor.
    conf_total: u16,
}

/// Bus time of one transaction carrying `len` data bytes, in us: token,
/// data packet with PID and CRC, handshake, bit stuffing and inter-packet
/// gaps at 12 Mbit/s.
pub fn slot_us(len: usize) -> u64 {
    // (3 + len + 3 + 1 bytes) * 8 bits * 7/6 stuffing / 12 bits per us,
    // plus about 3 us of turnaround.
    let bits = (7 + len as u64) * 8 * 7 / 6;
    3 + bits.div_ceil(12)
}

const CTRL_TIMEOUT_US: u64 = 5_000_000;

fn setup_pkt(bm: u8, req: u8, value: u16, index: u16, len: u16) -> [u8; 8] {
    let v = value.to_le_bytes();
    let i = index.to_le_bytes();
    let l = len.to_le_bytes();
    [bm, req, v[0], v[1], i[0], i[1], l[0], l[1]]
}

fn utf16_string(d: &[u8]) -> String {
    if d.len() < 2 {
        return String::new();
    }
    let n = (d[0] as usize).min(d.len());
    let units: Vec<u16> = d[2..n].chunks(2).filter(|c| c.len() == 2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    String::from_utf16_lossy(&units)
}

impl CdcHost {
    pub fn new() -> Self {
        Self {
            state: HostState::Detached,
            address: 5,
            line_coding: [0x00, 0xC2, 0x01, 0x00, 0, 0, 8],
            info: DeviceInfo::default(),
            stats: HostStats::default(),
            log: Vec::new(),
            error: None,
            ready_at_us: None,
            frame: 0,
            next_frame_us: 0,
            next_txn_us: 0,
            step: Step::DevDesc8,
            ctrl: None,
            tx: VecDeque::new(),
            rx: Vec::new(),
            in_toggle: 0,
            out_toggle: 0,
            out_turn: true,
            conf_total: 9,
        }
    }

    /// Queue bytes for the device, as a write to the tty.
    pub fn write(&mut self, data: &[u8]) {
        self.tx.extend(data.iter().copied());
    }

    /// Bytes still waiting to go out.
    pub fn tx_pending(&self) -> usize {
        self.tx.len()
    }

    /// Take everything the device sent so far, as a read from the tty.
    pub fn take_rx(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.rx)
    }

    pub fn is_ready(&self) -> bool {
        self.state == HostState::Ready
    }

    fn fail(&mut self, msg: String) {
        self.error = Some(msg);
        self.state = HostState::Failed;
        self.ctrl = None;
    }

    fn start_ctrl(&mut self, now: u64, setup: [u8; 8], out: Vec<u8>) {
        let in_len = if setup[0] & 0x80 != 0 { u16::from_le_bytes([setup[6], setup[7]]) as usize } else { 0 };
        let addr = match self.step {
            Step::DevDesc8 | Step::SetAddress => 0,
            _ => self.address,
        };
        self.ctrl = Some(Ctrl { addr, setup, out, out_pos: 0, in_len, got: Vec::new(), stage: Stage::Setup, toggle: 0, started_us: now });
    }

    /// One transaction of the control transfer in flight. Returns the bus
    /// time used and, when the transfer finished, its IN data.
    fn ctrl_txn(&mut self, dev: &mut UsbCtrl, now: u64) -> (u64, Option<Result<Vec<u8>, String>>) {
        let c = self.ctrl.as_mut().unwrap();
        if now.saturating_sub(c.started_us) > CTRL_TIMEOUT_US {
            let msg = format!("control transfer {:02x?} timed out in stage {:?}", c.setup, c.stage);
            return (0, Some(Err(msg)));
        }
        let mps = if self.info.max_packet0 == 0 { 64 } else { self.info.max_packet0 as usize };
        match c.stage {
            Stage::Setup => match dev.setup(c.addr, &c.setup) {
                UsbResp::Ack => {
                    c.toggle = 1;
                    c.stage = if c.setup[0] & 0x80 != 0 && c.in_len > 0 {
                        Stage::DataIn
                    } else if !c.out.is_empty() {
                        Stage::DataOut
                    } else {
                        Stage::StatusIn
                    };
                    (slot_us(8), None)
                }
                r => {
                    self.stats.timeouts += 1;
                    (slot_us(8), Some(Err(format!("SETUP {:02x?}: {r:?}", c.setup))))
                }
            },
            Stage::DataIn => match dev.in_token(c.addr, 0) {
                UsbResp::Data(d, pid) => {
                    let n = d.len();
                    if pid != c.toggle {
                        self.stats.seq_errors += 1;
                    } else {
                        c.got.extend_from_slice(&d);
                        c.toggle ^= 1;
                        if n < mps || c.got.len() >= c.in_len {
                            c.got.truncate(c.in_len);
                            c.stage = Stage::StatusOut;
                        }
                    }
                    (slot_us(n), None)
                }
                UsbResp::Nak => {
                    self.stats.naks += 1;
                    (slot_us(0), None)
                }
                UsbResp::Stall => {
                    self.stats.stalls += 1;
                    (slot_us(0), Some(Err(format!("STALL in data stage of {:02x?}", c.setup))))
                }
                r => (slot_us(0), Some(Err(format!("IN data stage of {:02x?}: {r:?}", c.setup)))),
            },
            Stage::DataOut => {
                let end = (c.out_pos + mps).min(c.out.len());
                let chunk = c.out[c.out_pos..end].to_vec();
                match dev.out_token(c.addr, 0, &chunk, c.toggle) {
                    UsbResp::Ack => {
                        c.out_pos = end;
                        c.toggle ^= 1;
                        if c.out_pos >= c.out.len() {
                            c.stage = Stage::StatusIn;
                        }
                        (slot_us(chunk.len()), None)
                    }
                    UsbResp::Nak => {
                        self.stats.naks += 1;
                        (slot_us(chunk.len()), None)
                    }
                    UsbResp::Stall => {
                        self.stats.stalls += 1;
                        (slot_us(chunk.len()), Some(Err(format!("STALL in data stage of {:02x?}", c.setup))))
                    }
                    r => (slot_us(chunk.len()), Some(Err(format!("OUT data stage of {:02x?}: {r:?}", c.setup)))),
                }
            }
            Stage::StatusOut => match dev.out_token(c.addr, 0, &[], 1) {
                UsbResp::Ack => (slot_us(0), Some(Ok(std::mem::take(&mut c.got)))),
                UsbResp::Nak => {
                    self.stats.naks += 1;
                    (slot_us(0), None)
                }
                r => (slot_us(0), Some(Err(format!("status OUT of {:02x?}: {r:?}", c.setup)))),
            },
            Stage::StatusIn => match dev.in_token(c.addr, 0) {
                UsbResp::Data(d, _) if d.is_empty() => (slot_us(0), Some(Ok(Vec::new()))),
                UsbResp::Data(d, _) => (slot_us(d.len()), Some(Err(format!("status IN of {:02x?} carried {} bytes", c.setup, d.len())))),
                UsbResp::Nak => {
                    self.stats.naks += 1;
                    (slot_us(0), None)
                }
                UsbResp::Stall => {
                    self.stats.stalls += 1;
                    (slot_us(0), Some(Err(format!("STALL in status stage of {:02x?}", c.setup))))
                }
                r => (slot_us(0), Some(Err(format!("status IN of {:02x?}: {r:?}", c.setup)))),
            },
        }
    }

    fn parse_config(&mut self, d: &[u8]) {
        let mut i = 0;
        let mut iface_class = 0u8;
        let mut iface_num = 0u8;
        if d.len() >= 6 {
            self.info.config_value = d[5];
        }
        while i + 1 < d.len() {
            let len = d[i] as usize;
            if len < 2 || i + len > d.len() {
                break;
            }
            match d[i + 1] {
                4 if len >= 9 => {
                    iface_num = d[i + 2];
                    iface_class = d[i + 5];
                    if iface_class == 0x02 && d[i + 6] == 0x02 {
                        self.info.comm_iface = iface_num;
                    }
                    if iface_class == 0x0A {
                        self.info.data_iface = iface_num;
                    }
                }
                5 if len >= 7 && iface_class == 0x0A && d[i + 3] & 3 == 2 => {
                    let addr = d[i + 2];
                    let mps = u16::from_le_bytes([d[i + 4], d[i + 5]]);
                    if addr & 0x80 != 0 {
                        self.info.ep_in = addr & 0x0F;
                        self.info.ep_in_size = mps;
                    } else {
                        self.info.ep_out = addr & 0x0F;
                        self.info.ep_out_size = mps;
                    }
                }
                _ => {}
            }
            let _ = iface_num;
            i += len;
        }
    }

    /// Start the request for the current step, or skip a step that has
    /// nothing to ask.
    fn issue_step(&mut self, now: u64) {
        let lang = 0x0409;
        match self.step {
            Step::DevDesc8 => self.start_ctrl(now, setup_pkt(0x80, 6, 0x0100, 0, 64), Vec::new()),
            Step::SetAddress => self.start_ctrl(now, setup_pkt(0x00, 5, self.address as u16, 0, 0), Vec::new()),
            Step::AddressWait(_) => {}
            Step::DevDesc => self.start_ctrl(now, setup_pkt(0x80, 6, 0x0100, 0, 18), Vec::new()),
            Step::ConfHeader => self.start_ctrl(now, setup_pkt(0x80, 6, 0x0200, 0, 9), Vec::new()),
            Step::ConfAll => {
                let total = self.conf_total;
                self.start_ctrl(now, setup_pkt(0x80, 6, 0x0200, 0, total), Vec::new())
            }
            Step::LangIds => self.start_ctrl(now, setup_pkt(0x80, 6, 0x0300, 0, 255), Vec::new()),
            Step::StrManufacturer | Step::StrProduct | Step::StrSerial => {
                let idx = match self.step {
                    Step::StrManufacturer => self.info.i_manufacturer,
                    Step::StrProduct => self.info.i_product,
                    _ => self.info.i_serial,
                };
                if idx == 0 {
                    self.advance(now, Vec::new());
                } else {
                    self.start_ctrl(now, setup_pkt(0x80, 6, 0x0300 | idx as u16, lang, 255), Vec::new())
                }
            }
            Step::SetConfig => self.start_ctrl(now, setup_pkt(0x00, 9, self.info.config_value as u16, 0, 0), Vec::new()),
            Step::LineCoding => {
                let lc = self.line_coding.to_vec();
                self.start_ctrl(now, setup_pkt(0x21, 0x20, 0, self.info.comm_iface as u16, 7), lc)
            }
            Step::LineState => self.start_ctrl(now, setup_pkt(0x21, 0x22, 0x0003, self.info.comm_iface as u16, 0), Vec::new()),
            Step::Done => {}
        }
    }

    /// A step's transfer finished with `d`: record it and move on.
    fn advance(&mut self, now: u64, d: Vec<u8>) {
        self.stats.control_transfers += 1;
        let next = match self.step {
            Step::DevDesc8 => {
                if d.len() >= 8 {
                    self.info.max_packet0 = d[7];
                }
                self.log.push(format!("device descriptor at address 0: {} bytes, bMaxPacketSize0 {}", d.len(), self.info.max_packet0));
                Step::SetAddress
            }
            Step::SetAddress => {
                self.log.push(format!("SET_ADDRESS {}", self.address));
                Step::AddressWait(now + 2_000)
            }
            Step::AddressWait(_) => Step::DevDesc,
            Step::DevDesc => {
                if d.len() >= 18 {
                    self.info.vid = u16::from_le_bytes([d[8], d[9]]);
                    self.info.pid = u16::from_le_bytes([d[10], d[11]]);
                    self.info.bcd_device = u16::from_le_bytes([d[12], d[13]]);
                    self.info.i_manufacturer = d[14];
                    self.info.i_product = d[15];
                    self.info.i_serial = d[16];
                }
                self.log.push(format!("device {:04x}:{:04x} bcdDevice {:04x}", self.info.vid, self.info.pid, self.info.bcd_device));
                Step::ConfHeader
            }
            Step::ConfHeader => {
                self.conf_total = if d.len() >= 4 { u16::from_le_bytes([d[2], d[3]]) } else { 9 };
                Step::ConfAll
            }
            Step::ConfAll => {
                self.parse_config(&d);
                self.log.push(format!(
                    "configuration {}: {} bytes, CDC comm interface {}, data interface {}, bulk IN EP{} {} bytes, bulk OUT EP{} {} bytes",
                    self.info.config_value, d.len(), self.info.comm_iface, self.info.data_iface,
                    self.info.ep_in, self.info.ep_in_size, self.info.ep_out, self.info.ep_out_size
                ));
                if self.info.ep_in == 0 || self.info.ep_out == 0 {
                    self.fail("no CDC data interface with bulk IN and OUT".into());
                    return;
                }
                Step::LangIds
            }
            Step::LangIds => Step::StrManufacturer,
            Step::StrManufacturer => {
                self.info.manufacturer = utf16_string(&d);
                Step::StrProduct
            }
            Step::StrProduct => {
                self.info.product = utf16_string(&d);
                Step::StrSerial
            }
            Step::StrSerial => {
                self.info.serial = utf16_string(&d);
                self.log.push(format!("strings: manufacturer '{}', product '{}', serial '{}'", self.info.manufacturer, self.info.product, self.info.serial));
                Step::SetConfig
            }
            Step::SetConfig => {
                self.log.push(format!("SET_CONFIGURATION {}", self.info.config_value));
                Step::LineCoding
            }
            Step::LineCoding => {
                let baud = u32::from_le_bytes([self.line_coding[0], self.line_coding[1], self.line_coding[2], self.line_coding[3]]);
                self.log.push(format!("SET_LINE_CODING {baud} baud"));
                Step::LineState
            }
            Step::LineState => {
                self.log.push("SET_CONTROL_LINE_STATE DTR RTS".into());
                Step::Done
            }
            Step::Done => Step::Done,
        };
        self.step = next;
        if next == Step::Done {
            self.state = HostState::Ready;
            self.ready_at_us = Some(now);
        } else if !matches!(next, Step::AddressWait(_)) {
            self.issue_step(now);
        }
    }

    fn bulk_txn(&mut self, dev: &mut UsbCtrl) -> u64 {
        let have_out = !self.tx.is_empty();
        if have_out && self.out_turn {
            self.out_turn = false;
            let mps = self.info.ep_out_size.clamp(8, 64) as usize;
            let n = self.tx.len().min(mps);
            let chunk: Vec<u8> = self.tx.iter().take(n).copied().collect();
            match dev.out_token(self.address, self.info.ep_out, &chunk, self.out_toggle) {
                UsbResp::Ack => {
                    self.tx.drain(..n);
                    self.out_toggle ^= 1;
                    self.stats.bulk_out_packets += 1;
                    self.stats.bulk_out_bytes += n as u64;
                }
                UsbResp::Nak => self.stats.naks += 1,
                UsbResp::Stall => self.stats.stalls += 1,
                _ => self.stats.timeouts += 1,
            }
            return slot_us(n);
        }
        self.out_turn = true;
        match dev.in_token(self.address, self.info.ep_in) {
            UsbResp::Data(d, pid) => {
                let n = d.len();
                if pid != self.in_toggle {
                    self.stats.seq_errors += 1;
                } else {
                    self.in_toggle ^= 1;
                    self.stats.bulk_in_packets += 1;
                    self.stats.bulk_in_bytes += n as u64;
                    self.rx.extend_from_slice(&d);
                }
                slot_us(n)
            }
            UsbResp::Nak => {
                self.stats.naks += 1;
                slot_us(0)
            }
            UsbResp::Stall => {
                self.stats.stalls += 1;
                slot_us(0)
            }
            _ => {
                self.stats.timeouts += 1;
                slot_us(0)
            }
        }
    }

    /// Advance the host to `now_us`. Returns true when it touched the
    /// device, so the caller can refresh the USB interrupt line.
    pub fn poll(&mut self, dev: &mut UsbCtrl, now_us: u64) -> bool {
        let mut touched = false;
        match self.state {
            HostState::Detached => {
                if dev.device_attached() {
                    self.state = HostState::Debounce(now_us + 100_000);
                }
                return false;
            }
            HostState::Debounce(t) => {
                if !dev.device_attached() {
                    self.state = HostState::Detached;
                } else if now_us >= t {
                    dev.bus_reset();
                    self.state = HostState::Reset(now_us + 10_000);
                    touched = true;
                }
                return touched;
            }
            HostState::Reset(t) => {
                if now_us < t {
                    return false;
                }
                self.state = HostState::Enumerating;
                self.next_frame_us = now_us;
                self.next_txn_us = now_us;
                self.step = Step::DevDesc8;
                self.issue_step(now_us);
            }
            HostState::Failed => return false,
            _ => {}
        }
        if !dev.device_attached() {
            self.state = HostState::Detached;
            return false;
        }
        // SOF at each frame start.
        while now_us >= self.next_frame_us {
            self.frame = (self.frame + 1) & 0x7FF;
            dev.sof(self.frame);
            touched = true;
            let frame_start = self.next_frame_us;
            self.next_frame_us += 1_000;
            if self.next_txn_us < frame_start + 1 {
                self.next_txn_us = frame_start + 1;
            }
        }
        // Transactions due by now; never across a frame boundary.
        while self.next_txn_us <= now_us && self.next_txn_us < self.next_frame_us {
            if let Step::AddressWait(t) = self.step {
                if self.next_txn_us < t {
                    break;
                }
                self.advance(self.next_txn_us, Vec::new());
                self.stats.control_transfers -= 1;
                continue;
            }
            let t = self.next_txn_us;
            let used = if self.ctrl.is_some() {
                let (used, done) = self.ctrl_txn(dev, t);
                match done {
                    None => {}
                    Some(Ok(d)) => {
                        self.ctrl = None;
                        self.advance(t, d);
                    }
                    Some(Err(e)) => {
                        let step = self.step;
                        self.fail(format!("{step:?}: {e}"));
                        return true;
                    }
                }
                used
            } else if self.state == HostState::Ready {
                self.bulk_txn(dev)
            } else {
                break;
            };
            touched = true;
            self.next_txn_us += used.max(1);
        }
        touched
    }
}

impl Default for CdcHost {
    fn default() -> Self {
        Self::new()
    }
}
