use super::cdc_host::{CdcHost, HostState};
use super::*;

const CLR: u32 = 3;
const SET: u32 = 2;

/// Controller enabled in device mode with the pull-up on and a host
/// plugged in, as `dcd_init` leaves it.
fn attached() -> UsbCtrl {
    let mut u = UsbCtrl::new();
    u.set_host_present(true);
    u.write32(USB_PWR, USB_PWR_VBUS_DETECT | USB_PWR_VBUS_DETECT_OVERRIDE_EN, 0);
    u.write32(MAIN_CTRL, MAIN_CTRL_CONTROLLER_EN, 0);
    u.write32(SIE_CTRL, SIE_CTRL_EP0_INT_1BUF | SIE_CTRL_PULLUP_EN, 0);
    u.write32(INTE, INT_BUFF_STATUS | INT_BUS_RESET | INT_SETUP_REQ, 0);
    u
}

fn bc_off(ep: u8, dir: Dir) -> u32 {
    (DPRAM_BUF_CTRL + ep as usize * 8 + if dir == Dir::In { 0 } else { 4 }) as u32
}

fn ep_ctrl_off(ep: u8, dir: Dir) -> u32 {
    (DPRAM_EP_CTRL + (ep as usize - 1) * 8 + if dir == Dir::In { 0 } else { 4 }) as u32
}

/// Arm an IN buffer the way `prepare_ep_buffer` does.
fn arm_in(u: &mut UsbCtrl, ep: u8, buf: u32, data: &[u8], pid: u8) {
    for (i, b) in data.iter().enumerate() {
        u.write_dpram(buf + i as u32, *b as u32, 1);
    }
    let v = data.len() as u32 | BUF_AVAIL | BUF_FULL | BUF_RESET_SEL | if pid == 1 { BUF_DATA1_PID } else { 0 };
    u.write_dpram(bc_off(ep, Dir::In), v, 4);
}

fn arm_out(u: &mut UsbCtrl, ep: u8, max: u32, pid: u8) {
    let v = max | BUF_AVAIL | BUF_RESET_SEL | if pid == 1 { BUF_DATA1_PID } else { 0 };
    u.write_dpram(bc_off(ep, Dir::Out), v, 4);
}

#[test]
fn detached_controller_answers_nothing() {
    let mut u = UsbCtrl::new();
    u.set_host_present(true);
    assert!(!u.device_attached(), "no pull-up yet");
    assert_eq!(u.setup(0, &[0; 8]), UsbResp::Timeout);
    u.bus_reset();
    assert_eq!(u.read32(SIE_STATUS) & SIE_STATUS_BUS_RESET, 0);
    assert!(!u.irq_level());
}

#[test]
fn bus_reset_raises_irq_and_clears_through_clear_alias() {
    let mut u = attached();
    u.bus_reset();
    let st = u.read32(SIE_STATUS);
    assert_ne!(st & SIE_STATUS_BUS_RESET, 0);
    assert_ne!(st & SIE_STATUS_CONNECTED, 0);
    assert_ne!(st & SIE_STATUS_VBUS_DETECTED, 0);
    assert_eq!(u.read32(INTS), INT_BUS_RESET);
    assert!(u.irq_level());
    u.write32(SIE_STATUS, SIE_STATUS_BUS_RESET, CLR);
    assert_eq!(u.read32(INTS), 0);
    assert!(!u.irq_level());
    assert_ne!(u.read32(SIE_STATUS) & SIE_STATUS_CONNECTED, 0, "still connected");
}

#[test]
fn setup_lands_in_dpram_and_clears_stall_arm() {
    let mut u = attached();
    u.write32(EP_STALL_ARM, 3, 0);
    let pkt = [0x80, 6, 0, 1, 0, 0, 64, 0];
    assert_eq!(u.setup(0, &pkt), UsbResp::Ack);
    for (i, b) in pkt.iter().enumerate() {
        assert_eq!(u.read_dpram(i as u32, 1), *b as u32);
    }
    assert_ne!(u.read32(INTR) & INT_SETUP_REQ, 0);
    assert_eq!(u.read32(EP_STALL_ARM), 0);
    u.write32(SIE_STATUS, SIE_STATUS_SETUP_REC, CLR);
    assert_eq!(u.read32(INTR) & INT_SETUP_REQ, 0);
}

#[test]
fn ep0_in_naks_until_armed_then_sends_once() {
    let mut u = attached();
    assert_eq!(u.in_token(0, 0), UsbResp::Nak);
    arm_in(&mut u, 0, DPRAM_EP0_BUF0 as u32, &[1, 2, 3], 1);
    assert_eq!(u.in_token(0, 0), UsbResp::Data(vec![1, 2, 3], 1));
    let bc = u.read_dpram(bc_off(0, Dir::In), 4);
    assert_eq!(bc & (BUF_AVAIL | BUF_FULL), 0, "AVAIL and FULL clear after sending");
    assert_eq!(bc & BUF_LEN_MASK, 3, "LEN stays");
    assert_eq!(u.read32(BUFF_STATUS), 1, "EP0 IN is bit 0");
    assert_ne!(u.read32(INTS) & INT_BUFF_STATUS, 0);
    assert_eq!(u.in_token(0, 0), UsbResp::Nak, "buffer is spent");
    u.write32(BUFF_STATUS, 1, CLR);
    assert_eq!(u.read32(BUFF_STATUS), 0);
}

#[test]
fn ep0_without_int_1buf_completes_silently() {
    let mut u = attached();
    u.write32(SIE_CTRL, SIE_CTRL_EP0_INT_1BUF, CLR);
    arm_in(&mut u, 0, DPRAM_EP0_BUF0 as u32, &[9], 0);
    assert_eq!(u.in_token(0, 0), UsbResp::Data(vec![9], 0));
    assert_eq!(u.read32(BUFF_STATUS), 0);
}

#[test]
fn bulk_out_writes_buffer_and_sets_len_full() {
    let mut u = attached();
    u.write_dpram(ep_ctrl_off(2, Dir::Out), EP_CTRL_ENABLE | EP_CTRL_INT_PER_BUFFER | (2 << 26) | 0x180, 4);
    assert_eq!(u.out_token(0, 2, &[7; 10], 0), UsbResp::Nak, "not armed");
    arm_out(&mut u, 2, 64, 0);
    assert_eq!(u.out_token(0, 2, &[0xAB, 0xCD], 0), UsbResp::Ack);
    assert_eq!(u.read_dpram(0x180, 2), 0xCDAB);
    let bc = u.read_dpram(bc_off(2, Dir::Out), 4);
    assert_eq!(bc & BUF_LEN_MASK, 2);
    assert_ne!(bc & BUF_FULL, 0);
    assert_eq!(bc & BUF_AVAIL, 0);
    assert_eq!(u.read32(BUFF_STATUS), 1 << 5, "EP2 OUT is bit 5");
    assert_eq!(u.out_token(0, 2, &[1], 1), UsbResp::Nak, "FULL until firmware re-arms");
}

#[test]
fn out_larger_than_len_is_refused() {
    let mut u = attached();
    u.write_dpram(ep_ctrl_off(1, Dir::Out), EP_CTRL_ENABLE | EP_CTRL_INT_PER_BUFFER | 0x180, 4);
    arm_out(&mut u, 1, 4, 0);
    assert_eq!(u.out_token(0, 1, &[0; 5], 0), UsbResp::Nak);
    assert_eq!(u.out_token(0, 1, &[0; 4], 0), UsbResp::Ack);
}

#[test]
fn out_with_wrong_pid_is_acked_and_dropped() {
    let mut u = attached();
    u.write_dpram(ep_ctrl_off(1, Dir::Out), EP_CTRL_ENABLE | EP_CTRL_INT_PER_BUFFER | 0x180, 4);
    arm_out(&mut u, 1, 64, 1);
    assert_eq!(u.out_token(0, 1, &[5], 0), UsbResp::Ack);
    assert_eq!(u.read32(BUFF_STATUS), 0, "nothing delivered");
    assert_ne!(u.read32(SIE_STATUS) & SIE_STATUS_DATA_SEQ_ERROR, 0);
    assert_eq!(u.stats.seq_errors, 1);
    assert_eq!(u.out_token(0, 1, &[5], 1), UsbResp::Ack);
    assert_eq!(u.read32(BUFF_STATUS), 1 << 3);
}

#[test]
fn double_buffered_in_alternates_and_interrupts_per_pair() {
    let mut u = attached();
    u.write_dpram(
        ep_ctrl_off(1, Dir::In),
        EP_CTRL_ENABLE | EP_CTRL_DOUBLE_BUFFERED | EP_CTRL_INT_PER_DOUBLE_BUFFER | (2 << 26) | 0x200,
        4,
    );
    for i in 0..64 {
        u.write_dpram(0x200 + i, 0x10, 1);
    }
    for i in 0..8 {
        u.write_dpram(0x240 + i, 0x20, 1);
    }
    let half0 = 64 | BUF_AVAIL | BUF_FULL | BUF_RESET_SEL;
    let half1 = 8 | BUF_AVAIL | BUF_FULL | BUF_DATA1_PID | BUF_LAST;
    u.write_dpram(bc_off(1, Dir::In), half0 | (half1 << 16), 4);
    assert_eq!(u.in_token(0, 1), UsbResp::Data(vec![0x10; 64], 0));
    assert_eq!(u.read32(BUFF_STATUS), 0, "first of the pair");
    assert_eq!(u.in_token(0, 1), UsbResp::Data(vec![0x20; 8], 1));
    assert_eq!(u.read32(BUFF_STATUS), 1 << 2, "EP1 IN is bit 2");
    assert_eq!(u.read32(BUFF_CPU_SHOULD_HANDLE), 1 << 2, "buffer 1 finished last");
    let bc = u.read_dpram(bc_off(1, Dir::In), 4);
    assert_eq!(bc & 0x3FF, 64);
    assert_eq!((bc >> 16) & 0x3FF, 8);
}

#[test]
fn disabled_endpoint_and_wrong_address_time_out() {
    let mut u = attached();
    assert_eq!(u.in_token(0, 3), UsbResp::Timeout, "EP3 not enabled");
    u.write32(ADDR_ENDP, 9, 0);
    assert_eq!(u.in_token(0, 0), UsbResp::Timeout, "device now at 9");
    assert_eq!(u.in_token(9, 0), UsbResp::Nak);
}

#[test]
fn ep0_stall_needs_arm() {
    let mut u = attached();
    u.write_dpram(bc_off(0, Dir::In), BUF_STALL, 4);
    assert_eq!(u.in_token(0, 0), UsbResp::Nak, "STALL without EP_STALL_ARM is a NAK");
    u.write32(EP_STALL_ARM, 1, SET);
    assert_eq!(u.in_token(0, 0), UsbResp::Stall);
    u.setup(0, &[0; 8]);
    assert_eq!(u.in_token(0, 0), UsbResp::Nak, "SETUP clears the arm");
}

#[test]
fn sof_latch_clears_on_sof_rd_read() {
    let mut u = attached();
    u.write32(INTE, INT_DEV_SOF, SET);
    u.sof(0x123);
    assert_ne!(u.read32(INTS) & INT_DEV_SOF, 0);
    assert_eq!(u.read32(SOF_RD), 0x123);
    assert_eq!(u.read32(INTS) & INT_DEV_SOF, 0);
}

#[test]
fn ep_abort_completes_at_once() {
    let mut u = attached();
    u.write32(EP_ABORT, 3, SET);
    assert_eq!(u.read32(EP_ABORT_DONE) & 3, 3);
    u.write32(EP_ABORT_DONE, 3, CLR);
    u.write32(EP_ABORT, 3, CLR);
    assert_eq!(u.read32(EP_ABORT_DONE), 0);
}

#[test]
fn reset_keeps_host_and_data_buffers() {
    let mut u = attached();
    u.write_dpram(0x400, 0x55, 1);
    u.write_dpram(0x80, 0xFFFF, 4);
    u.reset();
    assert_eq!(u.read32(MAIN_CTRL), 0x4, "RP2350 reset value");
    assert_eq!(u.read32(SIE_CTRL), 0x8000);
    assert_eq!(u.read_dpram(0x400, 1), 0x55);
    assert_eq!(u.read_dpram(0x80, 4), 0);
    u.write32(MAIN_CTRL, MAIN_CTRL_CONTROLLER_EN, 0);
    u.write32(SIE_CTRL, SIE_CTRL_PULLUP_EN, 0);
    assert!(u.device_attached(), "host still plugged in");
}

// ---------------------------------------------------------------------
// A small device stack in the style of TinyUSB's dcd_rp2040, so the host
// can be run end to end: control requests answered from EP0 buffer 0,
// bulk OUT on EP2 echoed back on bulk IN EP2, single buffered.

const DEV_DESC: [u8; 18] = [18, 1, 0x00, 0x02, 0xEF, 0x02, 0x01, 64, 0x8A, 0x2E, 0x0A, 0x00, 0x00, 0x01, 1, 2, 3, 1];

fn conf_desc() -> Vec<u8> {
    let mut d = vec![9, 2, 0, 0, 2, 1, 0, 0x80, 0x32];
    d.extend_from_slice(&[9, 4, 0, 0, 1, 0x02, 0x02, 0x00, 0]);
    d.extend_from_slice(&[7, 5, 0x81, 3, 8, 0, 16]);
    d.extend_from_slice(&[9, 4, 1, 0, 2, 0x0A, 0x00, 0x00, 0]);
    d.extend_from_slice(&[7, 5, 0x02, 2, 64, 0, 0]);
    d.extend_from_slice(&[7, 5, 0x82, 2, 64, 0, 0]);
    let n = d.len() as u16;
    d[2..4].copy_from_slice(&n.to_le_bytes());
    d
}

fn string_desc(s: &str) -> Vec<u8> {
    let mut d = vec![0, 3];
    for u in s.encode_utf16() {
        d.extend_from_slice(&u.to_le_bytes());
    }
    d[0] = d.len() as u8;
    d
}

#[derive(Default)]
struct MiniDev {
    /// IN data still to send on EP0, and the next PID.
    ep0_in: Vec<u8>,
    ep0_pid: u8,
    pending_addr: Option<u8>,
    configured: bool,
    dtr: bool,
    line_coding: Vec<u8>,
    expect_out_data: bool,
    echo: Vec<u8>,
    out_pid: u8,
    in_pid: u8,
    in_busy: bool,
}

impl MiniDev {
    fn send_ep0(&mut self, u: &mut UsbCtrl) {
        let n = self.ep0_in.len().min(64);
        let chunk: Vec<u8> = self.ep0_in.drain(..n).collect();
        arm_in(u, 0, DPRAM_EP0_BUF0 as u32, &chunk, self.ep0_pid);
        self.ep0_pid ^= 1;
    }

    fn open_bulk(&mut self, u: &mut UsbCtrl) {
        u.write_dpram(ep_ctrl_off(2, Dir::Out), EP_CTRL_ENABLE | EP_CTRL_INT_PER_BUFFER | (2 << 26) | 0x180, 4);
        u.write_dpram(ep_ctrl_off(2, Dir::In), EP_CTRL_ENABLE | EP_CTRL_INT_PER_BUFFER | (2 << 26) | 0x1C0, 4);
        arm_out(u, 2, 64, 0);
    }

    /// The interrupt handler.
    fn service(&mut self, u: &mut UsbCtrl) {
        let ints = u.read32(INTS);
        if ints & INT_BUS_RESET != 0 {
            u.write32(ADDR_ENDP, 0, 0);
            u.write32(SIE_STATUS, SIE_STATUS_BUS_RESET, CLR);
        }
        if ints & INT_BUFF_STATUS != 0 {
            let bs = u.read32(BUFF_STATUS);
            u.write32(BUFF_STATUS, bs, CLR);
            if bs & 1 != 0 {
                // EP0 IN done: more data, or the status stage it was.
                if let Some(a) = self.pending_addr.take() {
                    u.write32(ADDR_ENDP, a as u32, 0);
                } else if !self.ep0_in.is_empty() {
                    self.send_ep0(u);
                }
            }
            if bs & 2 != 0 && self.expect_out_data {
                let bc = u.read_dpram(bc_off(0, Dir::Out), 4);
                let n = bc & BUF_LEN_MASK;
                self.line_coding = (0..n).map(|i| u.read_dpram(DPRAM_EP0_BUF0 as u32 + i, 1) as u8).collect();
                self.expect_out_data = false;
                arm_in(u, 0, DPRAM_EP0_BUF0 as u32, &[], 1);
            }
            if bs & (1 << 5) != 0 {
                let bc = u.read_dpram(bc_off(2, Dir::Out), 4);
                let n = bc & BUF_LEN_MASK;
                for i in 0..n {
                    self.echo.push(u.read_dpram(0x180 + i, 1) as u8);
                }
                self.out_pid ^= 1;
                arm_out(u, 2, 64, self.out_pid);
            }
            if bs & (1 << 4) != 0 {
                self.in_busy = false;
            }
        }
        if ints & INT_SETUP_REQ != 0 {
            let s: Vec<u8> = (0..8).map(|i| u.read_dpram(i, 1) as u8).collect();
            u.write32(SIE_STATUS, SIE_STATUS_SETUP_REC, CLR);
            let value = u16::from_le_bytes([s[2], s[3]]);
            let len = u16::from_le_bytes([s[6], s[7]]) as usize;
            self.ep0_pid = 1;
            match (s[0], s[1]) {
                (0x80, 6) => {
                    let mut d = match value >> 8 {
                        1 => DEV_DESC.to_vec(),
                        2 => conf_desc(),
                        3 => match value & 0xFF {
                            0 => vec![4, 3, 0x09, 0x04],
                            1 => string_desc("Raspberry Pi"),
                            2 => string_desc("Pico"),
                            _ => string_desc("E66164084319392A"),
                        },
                        _ => Vec::new(),
                    };
                    d.truncate(len);
                    self.ep0_in = d;
                    self.send_ep0(u);
                    // OUT status buffer.
                    arm_out(u, 0, 0, 1);
                }
                (0x00, 5) => {
                    self.pending_addr = Some(value as u8);
                    arm_in(u, 0, DPRAM_EP0_BUF0 as u32, &[], 1);
                }
                (0x00, 9) => {
                    self.configured = true;
                    self.open_bulk(u);
                    arm_in(u, 0, DPRAM_EP0_BUF0 as u32, &[], 1);
                }
                (0x21, 0x20) => {
                    self.expect_out_data = true;
                    arm_out(u, 0, 64, 1);
                }
                (0x21, 0x22) => {
                    self.dtr = value & 1 != 0;
                    arm_in(u, 0, DPRAM_EP0_BUF0 as u32, &[], 1);
                }
                _ => {
                    u.write32(EP_STALL_ARM, 3, SET);
                    u.write_dpram(bc_off(0, Dir::In), BUF_STALL, 4);
                }
            }
        }
    }

    /// Main loop: echo what arrived.
    fn task(&mut self, u: &mut UsbCtrl) {
        if !self.in_busy && !self.echo.is_empty() {
            let n = self.echo.len().min(64);
            let d: Vec<u8> = self.echo.drain(..n).collect();
            arm_in(u, 2, 0x1C0, &d, self.in_pid);
            self.in_pid ^= 1;
            self.in_busy = true;
        }
    }
}

fn run_host(u: &mut UsbCtrl, dev: &mut MiniDev, host: &mut CdcHost, from_us: u64, to_us: u64) {
    let mut t = from_us;
    while t < to_us {
        host.poll(u, t);
        while u.irq_level() {
            dev.service(u);
        }
        dev.task(u);
        t += 5;
    }
}

#[test]
fn cdc_host_enumerates_and_echoes_through_a_device_stack() {
    let mut u = attached();
    let mut dev = MiniDev::default();
    let mut host = CdcHost::new();
    run_host(&mut u, &mut dev, &mut host, 0, 300_000);
    assert_eq!(host.state, HostState::Ready, "error: {:?}, log {:?}", host.error, host.log);
    assert_eq!(u.dev_addr(), host.address);
    assert!(dev.configured && dev.dtr);
    assert_eq!(dev.line_coding, host.line_coding.to_vec());
    assert_eq!(host.info.vid, 0x2E8A);
    assert_eq!(host.info.pid, 0x000A);
    assert_eq!(host.info.product, "Pico");
    assert_eq!(host.info.manufacturer, "Raspberry Pi");
    assert_eq!(host.info.serial, "E66164084319392A");
    assert_eq!((host.info.ep_in, host.info.ep_out), (2, 2));
    assert_eq!(host.info.comm_iface, 0);
    let ready = host.ready_at_us.unwrap();
    assert!(ready >= 110_000, "debounce and reset come first: {ready}");

    let msg: Vec<u8> = (0..200u32).map(|i| (i * 7) as u8).collect();
    host.write(&msg);
    run_host(&mut u, &mut dev, &mut host, 300_000, 320_000);
    assert_eq!(host.take_rx(), msg);
    assert_eq!(host.tx_pending(), 0);
    assert_eq!(host.stats.seq_errors, 0);
    assert_eq!(u.stats.seq_errors, 0);
    assert_eq!(host.stats.bulk_out_packets, 4, "200 bytes = 3 x 64 + 8");
    assert!(u.stats.sofs >= 200, "a SOF every ms");
}

#[test]
fn cdc_host_fails_on_a_stalled_request() {
    let mut u = attached();
    let mut dev = MiniDev::default();
    let mut host = CdcHost::new();
    // A device that stalls everything.
    let mut t = 0;
    while t < 300_000 {
        host.poll(&mut u, t);
        if u.irq_level() {
            let ints = u.read32(INTS);
            if ints & INT_BUS_RESET != 0 {
                u.write32(SIE_STATUS, SIE_STATUS_BUS_RESET, CLR);
            }
            if ints & INT_SETUP_REQ != 0 {
                u.write32(SIE_STATUS, SIE_STATUS_SETUP_REC, CLR);
                u.write32(EP_STALL_ARM, 3, SET);
                u.write_dpram(bc_off(0, Dir::In), BUF_STALL, 4);
            }
        }
        t += 5;
    }
    let _ = &mut dev;
    assert_eq!(host.state, HostState::Failed);
    assert!(host.error.as_deref().unwrap_or("").contains("STALL"), "{:?}", host.error);
}

#[test]
fn slot_time_matches_full_speed() {
    let t = cdc_host::slot_us(64);
    assert!((50..=65).contains(&t), "64-byte packet about 57 us at 12 Mbit/s, got {t}");
    assert!(cdc_host::slot_us(0) < 10);
}
