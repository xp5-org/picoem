// Emulator setup and VGA pin decode, taken from ../picoem_ntsc_sim/src/main.rs.
// Included by main.rs; the shared copy belongs in a picoem session crate.

// ---------------------------------------------------------------- constants

const SYS_HZ: u64 = 315_000_000;
const VGA_BASE_PIN: u32 = 6;
const VGA_HSYNC_BIT: u32 = 6; // GPIO12, active low
const VGA_VSYNC_BIT: u32 = 7; // GPIO13, active low
const VGA_PIXEL_HZ: f64 = 25_175_000.0;
const VGA_LINE_SAMPLES: usize = 800;
const VGA_TOTAL_LINES: usize = 525;
const VGA_HSYNC_SAMPLES: usize = 96;
const VGA_VSYNC_LINES: usize = 2;
const VGA_ACTIVE_OFFSET: usize = 144;
const VGA_ACTIVE_W: usize = 640;
const VGA_ACTIVE_LINES: usize = 480;
/// The first active line follows vsync start by 525 - 490 lines.
const VGA_VSYNC_TO_ACTIVE: usize = VGA_TOTAL_LINES - 490;

const NTSC_PIN: u32 = 28;
/// PWM wrap 11 at clkdiv 2: one sample = 22 sys cycles = 4 x fsc.
const NTSC_SAMPLE_CYCLES: u64 = 22;
const NTSC_LINE_SAMPLES: u64 = 908;
const NTSC_TOTAL_LINES: usize = 262;
const NTSC_HSYNC_SAMPLES: u64 = 68;

const BX_LR: [u8; 2] = [0x70, 0x47];

// --------------------------------------------------------------- ELF symbols

#[derive(Clone, Copy)]
struct Sym {
    addr: u32,
    size: u32,
}

fn u16le(b: &[u8], o: usize) -> u32 {
    u16::from_le_bytes([b[o], b[o + 1]]) as u32
}
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Minimal ELF32-LE .symtab reader with no dependencies.
fn read_symbols(elf: &[u8]) -> Result<HashMap<String, Sym>, String> {
    if elf.len() < 0x34 || &elf[0..4] != b"\x7fELF" || elf[4] != 1 || elf[5] != 1 {
        return Err("not an ELF32 little-endian file".into());
    }
    let shoff = u32le(elf, 0x20) as usize;
    let shentsize = u16le(elf, 0x2e) as usize;
    let shnum = u16le(elf, 0x30) as usize;
    let mut out = HashMap::new();
    for i in 0..shnum {
        let sh = shoff + i * shentsize;
        if u32le(elf, sh + 4) != 2 {
            continue;
        }
        let off = u32le(elf, sh + 16) as usize;
        let size = u32le(elf, sh + 20) as usize;
        let link = u32le(elf, sh + 24) as usize;
        let stroff = u32le(elf, shoff + link * shentsize + 16) as usize;
        for s in 0..size / 16 {
            let e = off + s * 16;
            let name_off = u32le(elf, e) as usize;
            let end = elf[stroff + name_off..].iter().position(|&c| c == 0).unwrap_or(0);
            let name = String::from_utf8_lossy(&elf[stroff + name_off..stroff + name_off + end]).to_string();
            if !name.is_empty() {
                out.insert(name, Sym { addr: u32le(elf, e + 4) & !1, size: u32le(elf, e + 8) });
            }
        }
    }
    if out.is_empty() {
        return Err("no .symtab in ELF".into());
    }
    Ok(out)
}

fn fail(msg: &str) -> ! {
    println!("RESULT: FAIL ({msg})");
    exit(1);
}

fn fnv(data: impl IntoIterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

// ----------------------------------------------------------------- emulator

struct Sim {
    emu: Emulator,
    speed: Vec<(String, std::time::Duration, u64, [u64; 5])>,
    phase: &'static str,
    core1_buf: Vec<u32>,
    core1_running: bool,
}

impl Sim {
    /// Stand-in for the bootrom's core1 wait loop: echo each FIFO word back
    /// to core0 and start core1 when the {0,0,1,vtor,sp,entry} sequence ends.
    fn core1_handshake(&mut self) {
        if self.core1_running || self.emu.bus.sio.fifo_0to1_snapshot().is_empty() {
            return;
        }
        while !self.emu.bus.sio.fifo_0to1_snapshot().is_empty() {
            let w = self.emu.bus.sio.read32(0x58, 1);
            self.emu.bus.sio.write32(0x54, w, 1);
            self.emu.bus.atomics.set_event_flag(0);
            if w == 0 && self.core1_buf.len() >= 3 {
                self.core1_buf.clear();
            }
            self.core1_buf.push(w);
            if self.core1_buf.len() == 6 && self.core1_buf[..3] == [0, 0, 1] {
                let (vt, sp, entry) = (self.core1_buf[3], self.core1_buf[4], self.core1_buf[5]);
                let c = self.emu.core_mut(1);
                c.regs.faultmask = 0;
                c.regs.primask = 0;
                c.ppb.vtor = vt;
                c.regs.r[13] = sp;
                c.regs.msp = sp;
                c.regs.set_pc(entry & !1);
                c.wake();
                self.core1_running = true;
            }
        }
    }

    fn exec_counters(&mut self) -> [u64; 5] {
        let (a, b) = (self.emu.core(0).counters.clone(), self.emu.core(1).counters.clone());
        [a.decode_execute_cycles, b.decode_execute_cycles, a.instructions, b.instructions,
         a.idle_skipped_cycles + b.idle_skipped_cycles]
    }

    /// Run until `stop` says so or `cycles` pass, in `q`-cycle quanta.
    fn run_for<F: FnMut(&mut Sim) -> bool>(&mut self, cycles: u64, q: u32, mut each: F) {
        self.emu.step_quantum = q;
        let t0 = std::time::Instant::now();
        let c0 = self.emu.cycles();
        let x0 = self.exec_counters();
        let end = c0 + cycles;
        while self.emu.cycles() < end {
            self.emu.run(q as u64).unwrap();
            self.core1_handshake();
            if each(self) {
                break;
            }
        }
        let key = format!("{} q{q}", self.phase);
        let dc = self.emu.cycles() - c0;
        let x1 = self.exec_counters();
        let dx = [x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2], x1[3] - x0[3], x1[4] - x0[4]];
        match self.speed.iter_mut().find(|e| e.0 == key) {
            Some(e) => { e.1 += t0.elapsed(); e.2 += dc; for i in 0..5 { e.3[i] += dx[i]; } }
            None => self.speed.push((key, t0.elapsed(), dc, dx)),
        }
    }
}

fn build_sim(fw_in: &[u8], rom: &[u8], syms: &HashMap<String, Sym>) -> Sim {
    let mut fw = fw_in.to_vec();
    // The bootrom writes the BOOTRAM lock status, and the harness skips the bootrom.
    let s = syms.get("runtime_init_bootrom_locking_enable")
        .copied()
        .unwrap_or_else(|| fail("symbol runtime_init_bootrom_locking_enable not in ELF"));
    if s.size < 2 {
        fail("runtime_init_bootrom_locking_enable too small to patch");
    }
    let off = (s.addr - 0x1000_0000) as usize;
    fw[off..off + 2].copy_from_slice(&BX_LR);

    let mut emu = Emulator::new(Config::default());
    emu.load_bootrom(rom);
    emu.load_flash(&fw);
    emu.reset();
    emu.core_mut(1).halt();
    emu.bus.atomics.rcp_salt_set(0, 0x1357_9bdf);
    emu.bus.atomics.rcp_salt_set(1, 0x2468_ace0);
    for i in 0..2 {
        let c = emu.core_mut(i);
        c.ppb.cpacr |= 0x0000_CD03;
        c.regs.faultmask = 0;
        c.regs.primask = 0;
    }
    let sp = u32::from_le_bytes(fw[0..4].try_into().unwrap());
    let pc = u32::from_le_bytes(fw[4..8].try_into().unwrap());
    let c = emu.core_mut(0);
    c.ppb.vtor = 0x1000_0000;
    c.regs.r[13] = sp;
    c.regs.msp = sp;
    c.regs.set_pc(pc & !1);
    Sim { emu, speed: Vec::new(), phase: "boot", core1_buf: Vec::new(), core1_running: false }
}

fn s_regs(sim: &Sim) -> String {
    let c = sim.emu.core(0);
    let r = &c.regs.r;
    format!("r0-r12 {:08x?} sp {:08x} lr {:08x} xpsr {:08x?}", &r[..13], r[13], r[14], c.regs.xpsr)
}

// ------------------------------------------------------------ VGA decoding

struct VgaReport {
    lines: usize,
    line_period_min: u64,
    line_period_max: u64,
    line_period_mean: f64,
    hsync_width_samples: (f64, f64),
    frames: Vec<VgaFrame>,
}

struct VgaFrame {
    vsync_cycle: u64,
    lines: usize,
    vsync_lines: usize,
    /// 800 x 525 RGB888 raster from the frame's vsync start.
    raster: Vec<u8>,
    colors_active: usize,
}

fn rgb222(v: u8) -> [u8; 3] {
    let c = |x: u8| (x & 3) * 85;
    [c(v >> 4), c(v >> 2), c(v)]
}

/// `pins[i]` = VGA byte on GPIO6..13 after cycle `t0 + i`.
fn decode_vga(pins: &[u8], t0: u64, sys_hz: f64) -> VgaReport {
    let hs = |b: u8| (b >> VGA_HSYNC_BIT) & 1 == 0;
    let vs = |b: u8| (b >> VGA_VSYNC_BIT) & 1 == 0;
    // Hsync start/end cycles, as indices into pins.
    let mut starts = Vec::new();
    let mut widths = Vec::new();
    let mut open: Option<usize> = None;
    for i in 1..pins.len() {
        let (a, b) = (hs(pins[i - 1]), hs(pins[i]));
        if b && !a {
            open = Some(i);
            starts.push(i);
        }
        if !b && a {
            if let Some(s) = open.take() {
                widths.push((i - s) as f64);
            }
        }
    }
    let periods: Vec<u64> = starts.windows(2).map(|w| (w[1] - w[0]) as u64).collect();
    let sample_cycles = sys_hz / VGA_PIXEL_HZ;
    let wmin = widths.iter().cloned().fold(f64::MAX, f64::min) / sample_cycles;
    let wmax = widths.iter().cloned().fold(0.0, f64::max) / sample_cycles;
    let mut rep = VgaReport {
        lines: starts.len(),
        line_period_min: periods.iter().copied().min().unwrap_or(0),
        line_period_max: periods.iter().copied().max().unwrap_or(0),
        line_period_mean: if periods.is_empty() { 0.0 } else { periods.iter().sum::<u64>() as f64 / periods.len() as f64 },
        hsync_width_samples: (wmin, wmax),
        frames: Vec::new(),
    };
    // Frames: each starts at the hsync that begins a vsync line.
    let mut vstarts = Vec::new();
    for (k, &s) in starts.iter().enumerate() {
        let mid = s + 4; // a few cycles into the line, past any skew
        if mid < pins.len() && vs(pins[mid]) && (k == 0 || !vs(pins[starts[k - 1] + 4])) && k > 0 {
            vstarts.push(k);
        }
    }
    for w in vstarts.windows(2) {
        let (k0, k1) = (w[0], w[1]);
        let mut vsync_lines = 0;
        let mut raster = vec![0u8; VGA_LINE_SAMPLES * VGA_TOTAL_LINES * 3];
        let mut colors = std::collections::HashSet::new();
        for (l, k) in (k0..k1).enumerate() {
            if vs(pins[starts[k] + 4]) {
                vsync_lines += 1;
            }
            if l >= VGA_TOTAL_LINES {
                continue;
            }
            for x in 0..VGA_LINE_SAMPLES {
                let at = starts[k] + ((x as f64 + 0.5) * sample_cycles) as usize;
                let b = if at < pins.len() { pins[at] } else { 0 };
                let o = (l * VGA_LINE_SAMPLES + x) * 3;
                raster[o..o + 3].copy_from_slice(&rgb222(b));
                let active = (VGA_VSYNC_TO_ACTIVE..VGA_VSYNC_TO_ACTIVE + VGA_ACTIVE_LINES).contains(&l)
                    && (VGA_ACTIVE_OFFSET..VGA_ACTIVE_OFFSET + VGA_ACTIVE_W).contains(&x);
                if active {
                    colors.insert(b & 0x3f);
                }
            }
        }
        rep.frames.push(VgaFrame {
            vsync_cycle: t0 + starts[k0] as u64,
            lines: k1 - k0,
            vsync_lines,
            raster,
            colors_active: colors.len(),
        });
    }
    rep
}


/// Capture `cycles` at q=1: VGA byte and the NTSC pin after every cycle.
fn capture(sim: &mut Sim, cycles: u64) -> (u64, Vec<u8>, Vec<bool>) {
    let t0 = sim.emu.cycles();
    let mut vga = Vec::with_capacity(cycles as usize);
    let mut ntsc = Vec::with_capacity(cycles as usize);
    sim.run_for(cycles, 1, |s| {
        let g = s.emu.gpio_read_all();
        vga.push((g >> VGA_BASE_PIN) as u8);
        ntsc.push((g >> NTSC_PIN) & 1 == 1);
        false
    });
    (t0 + 1, vga, ntsc)
}

// ----------------------------------------------------------- NTSC decoding

struct NtscReport {
    pwm_periods: usize,
    pwm_period_ok: bool,
    lines: usize,
    line_period_min: u64,
    line_period_max: u64,
    /// Hsync-to-hsync periods that are not a whole number of lines.
    off_grid: usize,
    hsync_samples: (u64, u64),
    frame_lines: Vec<usize>,
    /// One luma frame, 908 x 262 levels of 0..11, from the first vsync.
    luma: Option<Vec<u8>>,
    levels_seen: usize,
}

/// `pin[i]` = GPIO28 after cycle `t0 + i`. One PWM period is one sample: 22
/// cycles, the counter wrapping 0..10 at clkdiv 2. Its high time is the level.
fn decode_ntsc(pin: &[bool]) -> NtscReport {
    let mut rep = NtscReport {
        pwm_periods: 0, pwm_period_ok: false, lines: 0, line_period_min: 0, line_period_max: 0, off_grid: 0,
        hsync_samples: (0, 0), frame_lines: Vec::new(), luma: None, levels_seen: 0,
    };
    // Rising edges mark the start of every period with level > 0.
    let rises: Vec<usize> = (1..pin.len()).filter(|&i| pin[i] && !pin[i - 1]).collect();
    rep.pwm_periods = rises.len();
    if rises.len() < 100 {
        return rep;
    }
    // Period phase: rises fall on one residue mod 22 when the PWM runs.
    let phase = rises[0] % NTSC_SAMPLE_CYCLES as usize;
    rep.pwm_period_ok = rises.iter().all(|r| r % NTSC_SAMPLE_CYCLES as usize == phase);
    let mut levels = Vec::new();
    let mut i = phase;
    while i + NTSC_SAMPLE_CYCLES as usize <= pin.len() {
        let high = pin[i..i + NTSC_SAMPLE_CYCLES as usize].iter().filter(|&&b| b).count();
        levels.push((high / 2) as u8);
        i += NTSC_SAMPLE_CYCLES as usize;
    }
    let mut seen = [false; 16];
    for &l in &levels {
        seen[(l & 15) as usize] = true;
    }
    rep.levels_seen = seen.iter().filter(|&&s| s).count();
    // Sync = runs of level 0. Hsync ~68 samples, vsync pulses are much longer.
    let mut runs = Vec::new(); // start sample, length
    let mut s = None;
    for (k, &l) in levels.iter().enumerate() {
        match (l == 0, s) {
            (true, None) => s = Some(k),
            (false, Some(st)) => { runs.push((st, k - st)); s = None; }
            _ => {}
        }
    }
    let hs: Vec<&(usize, usize)> = runs.iter().filter(|r| r.1 >= 40 && r.1 <= 100).collect();
    rep.lines = hs.len();
    if hs.len() > 2 {
        let p: Vec<u64> = hs.windows(2).map(|w| (w[1].0 - w[0].0) as u64 * NTSC_SAMPLE_CYCLES).collect();
        rep.line_period_min = *p.iter().min().unwrap();
        rep.line_period_max = *p.iter().max().unwrap();
        rep.off_grid = p.iter().filter(|&&x| x % (NTSC_LINE_SAMPLES * NTSC_SAMPLE_CYCLES) != 0).count();
        rep.hsync_samples = (hs.iter().map(|r| r.1 as u64).min().unwrap(), hs.iter().map(|r| r.1 as u64).max().unwrap());
    }
    // Vertical sync: a line-long low run. Frames are counted from one to the next.
    let vs: Vec<usize> = runs.iter().filter(|r| r.1 as u64 > NTSC_LINE_SAMPLES / 2).map(|r| r.0).collect();
    let mut vfirst = Vec::new();
    for (j, &v) in vs.iter().enumerate() {
        if j == 0 || v - vs[j - 1] > (NTSC_LINE_SAMPLES * 20) as usize {
            vfirst.push(v);
        }
    }
    for w in vfirst.windows(2) {
        let n = (w[1] - w[0]) as f64 / NTSC_LINE_SAMPLES as f64;
        rep.frame_lines.push(n.round() as usize);
    }
    if let Some(&v0) = vfirst.first() {
        let n = NTSC_LINE_SAMPLES as usize * NTSC_TOTAL_LINES;
        if v0 + n <= levels.len() {
            rep.luma = Some(levels[v0..v0 + n].to_vec());
        }
    }
    rep
}

