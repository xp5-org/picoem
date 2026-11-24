//! QMI direct mode and the QSPI PSRAM on chip select 1.
//!
//! Direct mode is the QMI's serial port: firmware writes DIRECT_TX, the
//! QMI clocks the word out on the QSPI pins with CS0 or CS1 asserted, and
//! the word clocked back lands in the RX FIFO that DIRECT_RX pops. The
//! transfer completes at once here, so BUSY never reads set.
//!
//! Chip select follows DIRECT_CSR: ASSERT_CSnN holds a chip selected
//! across words, and AUTO_CSnN selects it for each word on its own. A
//! frame ends when the chip is deselected.
//!
//! The flash on CS0 answers the identification commands the SDK and boot
//! code send: 0x9F JEDEC ID, 0x4B unique ID, 0x05/0x35 status, 0x03 read.
//! Program and erase traffic is accepted and ignored; flash contents only
//! change through the emulator's flash loader.
//!
//! The PSRAM on CS1 is an APS6404L-style part. In direct mode it takes
//! 0x35 enter QPI, 0xF5 exit QPI, 0x66 0x99 reset, 0x9F read ID with
//! MFID 0x0D and KGD 0x5D after the three address bytes, 0x03 read and
//! 0x02 write. Its array is the memory behind the XIP M1 window, read and
//! written by [`QmiDirect::psram_read`] and [`QmiDirect::psram_write`].
//! Addresses past the capacity wrap, as on the real parts, which is how
//! firmware sizes them.

use std::collections::VecDeque;

pub const DIRECT_CSR: u32 = 0x000;
pub const DIRECT_TX: u32 = 0x004;
pub const DIRECT_RX: u32 = 0x008;

const CSR_EN: u32 = 1 << 0;
const CSR_BUSY: u32 = 1 << 1;
const CSR_ASSERT_CS0N: u32 = 1 << 2;
const CSR_ASSERT_CS1N: u32 = 1 << 3;
const CSR_AUTO_CS0N: u32 = 1 << 6;
const CSR_AUTO_CS1N: u32 = 1 << 7;
const CSR_TXFULL: u32 = 1 << 10;
const CSR_TXEMPTY: u32 = 1 << 11;
const CSR_RXEMPTY: u32 = 1 << 16;
const CSR_RXFULL: u32 = 1 << 17;
const CSR_RXLEVEL_LSB: u32 = 18;
/// Bits firmware writes and reads back: everything but the status bits.
const CSR_RW_MASK: u32 = !(CSR_BUSY | CSR_TXFULL | CSR_TXEMPTY | CSR_RXEMPTY | CSR_RXFULL | (0x7 << CSR_RXLEVEL_LSB) | (0x7 << 12));

const TX_DWIDTH: u32 = 1 << 18;
const TX_NOPUSH: u32 = 1 << 20;

const RX_FIFO_DEPTH: usize = 4;

/// APS6404L manufacturer and known-good-die ID bytes.
pub const PSRAM_MFID: u8 = 0x0D;
pub const PSRAM_KGD: u8 = 0x5D;

/// Winbond W25Q-style JEDEC ID returned by the CS0 flash.
const FLASH_JEDEC: [u8; 3] = [0xEF, 0x40, 0x16];

/// QSPI PSRAM on CS1.
pub struct Psram {
    mem: Vec<u8>,
    qpi: bool,
    reset_enabled: bool,
}

impl Psram {
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "PSRAM size must be a power of two");
        Self { mem: vec![0; size], qpi: false, reset_enabled: false }
    }

    pub fn size(&self) -> usize {
        self.mem.len()
    }

    pub fn qpi(&self) -> bool {
        self.qpi
    }

    /// The whole array, for harnesses that inspect what firmware keeps there.
    pub fn data(&self) -> &[u8] {
        &self.mem
    }

    #[inline]
    fn index(&self, offset: u32) -> usize {
        offset as usize & (self.mem.len() - 1)
    }

    /// One byte of a direct-mode frame. `pos` is its index in the frame,
    /// `frame` the bytes before it. Returns the byte the PSRAM drives.
    fn direct_byte(&mut self, frame: &[u8], pos: usize, b: u8) -> u8 {
        if pos == 0 {
            match b {
                0x35 => self.qpi = true,
                0xF5 => self.qpi = false,
                0x66 => {
                    self.reset_enabled = true;
                    return 0;
                }
                0x99 if self.reset_enabled => self.qpi = false,
                _ => {}
            }
            self.reset_enabled = false;
            return 0;
        }
        let addr = |f: &[u8]| ((f[1] as u32) << 16) | ((f[2] as u32) << 8) | f[3] as u32;
        match frame[0] {
            0x9F if !self.qpi => match pos {
                4 => PSRAM_MFID,
                5 => PSRAM_KGD,
                _ => 0,
            },
            0x03 if pos >= 4 => {
                let i = self.index(addr(frame) + (pos - 4) as u32);
                self.mem[i]
            }
            0x02 | 0x38 if pos >= 4 => {
                let i = self.index(addr(frame) + (pos - 4) as u32);
                self.mem[i] = b;
                0
            }
            _ => 0,
        }
    }
}

/// QMI direct-mode port with its two chip selects.
pub struct QmiDirect {
    csr: u32,
    rx: VecDeque<u16>,
    cs0_frame: Vec<u8>,
    cs1_frame: Vec<u8>,
    psram: Option<Psram>,
    /// Flash image seen by CS0 reads. Set by the bus from the loaded flash.
    flash_unique_id: [u8; 8],
}

impl Default for QmiDirect {
    fn default() -> Self {
        Self::new()
    }
}

impl QmiDirect {
    pub fn new() -> Self {
        Self {
            csr: 0,
            rx: VecDeque::new(),
            cs0_frame: Vec::new(),
            cs1_frame: Vec::new(),
            psram: None,
            flash_unique_id: [0xE6, 0x60, 0x58, 0x38, 0x83, 0x51, 0x2A, 0x2F],
        }
    }

    /// Reset the port. The PSRAM is board wiring and keeps its contents
    /// and mode: a chip reset of the RP2350 does not power-cycle it.
    pub fn reset(&mut self) {
        let psram = self.psram.take();
        *self = Self::new();
        self.psram = psram;
    }

    /// Fit a PSRAM of `size` bytes on CS1.
    pub fn attach_psram(&mut self, size: usize) {
        self.psram = Some(Psram::new(size));
    }

    pub fn psram(&self) -> Option<&Psram> {
        self.psram.as_ref()
    }

    pub fn psram_mut(&mut self) -> Option<&mut Psram> {
        self.psram.as_mut()
    }

    /// Memory-mapped PSRAM read through the XIP M1 window. None without a part.
    #[inline]
    pub fn psram_read(&self, offset: u32, size: u32) -> Option<u32> {
        let p = self.psram.as_ref()?;
        let mut v = 0u32;
        for k in 0..size {
            v |= (p.mem[p.index(offset + k)] as u32) << (8 * k);
        }
        Some(v)
    }

    /// Memory-mapped PSRAM write. False without a part.
    #[inline]
    pub fn psram_write(&mut self, offset: u32, size: u32, val: u32) -> bool {
        let Some(p) = self.psram.as_mut() else { return false };
        for k in 0..size {
            let i = p.index(offset + k);
            p.mem[i] = (val >> (8 * k)) as u8;
        }
        true
    }

    fn cs0_held(&self) -> bool {
        self.csr & CSR_ASSERT_CS0N != 0
    }

    fn cs1_held(&self) -> bool {
        self.csr & CSR_ASSERT_CS1N != 0
    }

    pub fn read32(&mut self, offset: u32) -> u32 {
        match offset {
            DIRECT_CSR => {
                let lvl = self.rx.len().min(7) as u32;
                let mut v = (self.csr & CSR_RW_MASK) | CSR_TXEMPTY | (lvl << CSR_RXLEVEL_LSB);
                if self.rx.is_empty() {
                    v |= CSR_RXEMPTY;
                }
                if self.rx.len() >= RX_FIFO_DEPTH {
                    v |= CSR_RXFULL;
                }
                v
            }
            DIRECT_RX => self.rx.pop_front().unwrap_or(0) as u32,
            _ => 0,
        }
    }

    /// Returns true when `offset` is a direct-mode register this model owns.
    pub fn owns(offset: u32) -> bool {
        matches!(offset, DIRECT_CSR | DIRECT_TX | DIRECT_RX)
    }

    pub fn write32(&mut self, offset: u32, val: u32) {
        match offset {
            DIRECT_CSR => {
                let was0 = self.cs0_held();
                let was1 = self.cs1_held();
                self.csr = val & CSR_RW_MASK;
                if was0 && !self.cs0_held() {
                    self.cs0_frame.clear();
                }
                if was1 && !self.cs1_held() {
                    self.cs1_frame.clear();
                }
                if self.csr & CSR_EN == 0 {
                    self.rx.clear();
                }
            }
            DIRECT_TX => self.tx(val),
            _ => {}
        }
    }

    fn tx(&mut self, val: u32) {
        if self.csr & CSR_EN == 0 {
            return;
        }
        let bytes: &[u8] = &if val & TX_DWIDTH != 0 {
            [(val >> 8) as u8, val as u8].to_vec()
        } else {
            [val as u8].to_vec()
        };
        let cs0 = self.cs0_held() || self.csr & CSR_AUTO_CS0N != 0;
        let cs1 = self.cs1_held() || self.csr & CSR_AUTO_CS1N != 0;
        let mut reply = 0u16;
        for &b in bytes {
            let mut r = 0xFFu8;
            if cs1 {
                if let Some(p) = self.psram.as_mut() {
                    let pos = self.cs1_frame.len();
                    r = p.direct_byte(&self.cs1_frame, pos, b);
                }
                self.cs1_frame.push(b);
            }
            if cs0 {
                r = self.flash_byte(b);
                self.cs0_frame.push(b);
            }
            reply = (reply << 8) | r as u16;
        }
        if !self.cs0_held() {
            self.cs0_frame.clear();
        }
        if !self.cs1_held() {
            self.cs1_frame.clear();
        }
        if val & TX_NOPUSH == 0 && self.rx.len() < RX_FIFO_DEPTH {
            self.rx.push_back(reply);
        }
    }

    fn flash_byte(&self, b: u8) -> u8 {
        let f = &self.cs0_frame;
        let pos = f.len();
        if pos == 0 {
            return 0xFF;
        }
        match f[0] {
            0x9F => FLASH_JEDEC.get(pos - 1).copied().unwrap_or(0xFF),
            0x4B if pos >= 5 => self.flash_unique_id.get(pos - 5).copied().unwrap_or(0xFF),
            0x05 | 0x35 | 0x15 => 0x00,
            _ => {
                let _ = b;
                0xFF
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xfer(q: &mut QmiDirect, tx: &[u8]) -> Vec<u8> {
        q.write32(DIRECT_CSR, q.csr | CSR_ASSERT_CS1N);
        let mut out = Vec::new();
        for &b in tx {
            q.write32(DIRECT_TX, b as u32);
            assert_eq!(q.read32(DIRECT_CSR) & CSR_BUSY, 0);
            out.push(q.read32(DIRECT_RX) as u8);
        }
        q.write32(DIRECT_CSR, q.csr & !CSR_ASSERT_CS1N);
        out
    }

    #[test]
    fn psram_reset_then_read_id() {
        let mut q = QmiDirect::new();
        q.attach_psram(8 << 20);
        q.write32(DIRECT_CSR, CSR_EN | (30 << 22));
        xfer(&mut q, &[0x35]);
        assert!(q.psram().unwrap().qpi());
        xfer(&mut q, &[0xF5]);
        xfer(&mut q, &[0x66]);
        xfer(&mut q, &[0x99]);
        let id = xfer(&mut q, &[0x9F, 0, 0, 0, 0xFF, 0xFF]);
        assert_eq!(&id[4..], &[PSRAM_MFID, PSRAM_KGD]);
    }

    #[test]
    fn auto_cs_nopush_enters_qpi() {
        let mut q = QmiDirect::new();
        q.attach_psram(8 << 20);
        q.write32(DIRECT_CSR, CSR_EN | CSR_AUTO_CS1N | (10 << 22));
        q.write32(DIRECT_TX, TX_NOPUSH | 0x35);
        assert!(q.psram().unwrap().qpi());
        assert_ne!(q.read32(DIRECT_CSR) & CSR_RXEMPTY, 0);
        assert_ne!(q.read32(DIRECT_CSR) & CSR_TXEMPTY, 0);
    }

    #[test]
    fn no_psram_reads_ones() {
        let mut q = QmiDirect::new();
        q.write32(DIRECT_CSR, CSR_EN);
        let id = xfer(&mut q, &[0x9F, 0, 0, 0, 0xFF, 0xFF]);
        assert_eq!(&id[4..], &[0xFF, 0xFF]);
    }

    #[test]
    fn window_wraps_at_capacity() {
        let mut q = QmiDirect::new();
        q.attach_psram(8 << 20);
        assert!(q.psram_write(0, 4, 0x1357_9BDF));
        assert!(q.psram_write(8 << 20, 4, 0x2468_ACE0));
        assert_eq!(q.psram_read(0, 4), Some(0x2468_ACE0));
        assert_eq!(q.psram_read(1, 2), Some(0x68AC));
    }

    #[test]
    fn direct_write_then_read_back() {
        let mut q = QmiDirect::new();
        q.attach_psram(1 << 20);
        q.write32(DIRECT_CSR, CSR_EN);
        xfer(&mut q, &[0x02, 0x00, 0x01, 0x00, 0xAA, 0x55]);
        let r = xfer(&mut q, &[0x03, 0x00, 0x01, 0x00, 0, 0]);
        assert_eq!(&r[4..], &[0xAA, 0x55]);
        assert_eq!(q.psram_read(0x100, 2), Some(0x55AA));
    }

    #[test]
    fn flash_jedec_id_on_cs0() {
        let mut q = QmiDirect::new();
        q.write32(DIRECT_CSR, CSR_EN | CSR_ASSERT_CS0N);
        let mut out = Vec::new();
        for b in [0x9Fu8, 0, 0, 0] {
            q.write32(DIRECT_TX, b as u32);
            out.push(q.read32(DIRECT_RX) as u8);
        }
        assert_eq!(&out[1..], &FLASH_JEDEC);
    }
}
