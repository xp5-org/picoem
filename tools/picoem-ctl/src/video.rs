//! Video inputs as data: which pins carry which signal, how resistors turn pins into an analog
//! level, and which timing the decoder expects. A decoder reads the logged pin edges, not a
//! sample per cycle.
//!
//! An analog channel is a resistor network. Pin i drives a node through R_i, the node ends in
//! a load R_L to ground, and a pin driven low pulls the node down through its resistor. With
//! G_i = 1/R_i the node voltage is V = Vdd * sum(G_i * duty_i) / (sum(G_i) + G_L), where duty_i
//! is the share of the sample window the pin spends high. The decoder works with the level
//! normalized to full scale, sum(G_i * duty_i) / sum(G_i), so the load only matters for the
//! voltage shown to the user.

use crate::json::{json_str, J};

// ------------------------------------------------------------------ timing

#[derive(Clone, Debug)]
pub struct Timing {
    pub name: String,
    /// RGB with separate sync: the dot clock, from which the cycles per sample follow.
    pub pixel_hz: f64,
    pub line_samples: usize,
    pub total_lines: usize,
    pub hsync_samples: usize,
    pub vsync_lines: usize,
    pub active_offset: usize,
    pub active_w: usize,
    pub active_lines: usize,
    pub vsync_to_active: usize,
    /// Composite: system cycles per sample, which is one PWM period.
    pub sample_cycles: u64,
    /// Composite: a level below this is sync.
    pub sync_thr: f64,
    /// Composite: the accepted hsync pulse length range in samples.
    pub hs_min: usize,
    pub hs_max: usize,
}

pub const TIMING_NAMES: &[&str] = &["vga_640x480_60", "ntsc_m_22"];

impl Timing {
    pub fn preset(name: &str) -> Option<Timing> {
        let base = Timing {
            name: name.into(),
            pixel_hz: 0.0,
            line_samples: 0,
            total_lines: 0,
            hsync_samples: 0,
            vsync_lines: 0,
            active_offset: 0,
            active_w: 0,
            active_lines: 0,
            vsync_to_active: 0,
            sample_cycles: 0,
            sync_thr: 0.0,
            hs_min: 0,
            hs_max: 0,
        };
        match name {
            "vga_640x480_60" => Some(Timing {
                pixel_hz: 25_175_000.0,
                line_samples: 800,
                total_lines: 525,
                hsync_samples: 96,
                vsync_lines: 2,
                active_offset: 144,
                active_w: 640,
                active_lines: 480,
                vsync_to_active: 35,
                ..base
            }),
            "ntsc_m_22" => Some(Timing {
                line_samples: 908,
                total_lines: 262,
                hsync_samples: 68,
                active_w: 908,
                active_lines: 262,
                sample_cycles: 22,
                sync_thr: 0.09,
                hs_min: 40,
                hs_max: 100,
                ..base
            }),
            _ => None,
        }
    }

    pub fn parse(j: &J) -> Result<Timing, String> {
        let (mut t, over) = match j {
            J::Str(n) => (Timing::preset(n).ok_or(format!("unknown timing preset {n}"))?, None),
            J::Obj(_) => {
                let n = j.str("preset").ok_or("timing object needs \"preset\" as its base")?;
                (Timing::preset(n).ok_or(format!("unknown timing preset {n}"))?, Some(j))
            }
            _ => return Err("timing is a preset name or an object".into()),
        };
        if let Some(o) = over {
            let set = |k: &str, v: &mut usize| {
                if let Some(n) = o.num(k) {
                    *v = n as usize;
                }
            };
            set("line_samples", &mut t.line_samples);
            set("total_lines", &mut t.total_lines);
            set("hsync_samples", &mut t.hsync_samples);
            set("vsync_lines", &mut t.vsync_lines);
            set("active_offset", &mut t.active_offset);
            set("active_w", &mut t.active_w);
            set("active_lines", &mut t.active_lines);
            set("vsync_to_active", &mut t.vsync_to_active);
            set("hs_min", &mut t.hs_min);
            set("hs_max", &mut t.hs_max);
            if let Some(n) = o.num("pixel_hz") {
                t.pixel_hz = n;
            }
            if let Some(n) = o.num("sample_cycles") {
                t.sample_cycles = n as u64;
            }
            if let Some(n) = o.num("sync_thr") {
                t.sync_thr = n;
            }
            t.name = format!("{} custom", t.name);
        }
        Ok(t)
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"preset\":{},\"pixel_hz\":{},\"line_samples\":{},\"total_lines\":{},\"hsync_samples\":{},\
             \"vsync_lines\":{},\"active_offset\":{},\"active_w\":{},\"active_lines\":{},\"vsync_to_active\":{},\
             \"sample_cycles\":{},\"sync_thr\":{},\"hs_min\":{},\"hs_max\":{}}}",
            json_str(self.name.split(' ').next().unwrap_or("")),
            self.pixel_hz, self.line_samples, self.total_lines, self.hsync_samples, self.vsync_lines,
            self.active_offset, self.active_w, self.active_lines, self.vsync_to_active,
            self.sample_cycles, self.sync_thr, self.hs_min, self.hs_max
        )
    }
}

// ----------------------------------------------------------------- channel

#[derive(Clone, Debug)]
pub struct Channel {
    pub name: String,
    /// GPIO and series resistor in ohms, one entry per pin.
    pub pins: Vec<(u8, f64)>,
    pub load_ohm: f64,
    pub vdd: f64,
    /// 0: the level at an instant. N: the mean over windows of N system cycles, which is how a
    /// PWM pin or an RC filter reads.
    pub window_cycles: u64,
}

impl Channel {
    pub fn parse(name: &str, j: &J) -> Result<Channel, String> {
        let arr = match j.get("pins") {
            Some(J::Arr(a)) if !a.is_empty() => a,
            _ => return Err(format!("channel {name} needs a non-empty \"pins\" list")),
        };
        let mut pins = Vec::new();
        for p in arr {
            let (pin, ohm) = (p.num("pin"), p.num("ohm"));
            match (pin, ohm) {
                (Some(pin), Some(ohm)) if (0.0..48.0).contains(&pin) && ohm > 0.0 => pins.push((pin as u8, ohm)),
                _ => return Err(format!("channel {name}: each pin is {{\"pin\": 0..47, \"ohm\": > 0}}")),
            }
        }
        Ok(Channel {
            name: name.into(),
            pins,
            load_ohm: j.num("load_ohm").unwrap_or(75.0),
            vdd: j.num("vdd").unwrap_or(3.3),
            window_cycles: j.num("window_cycles").unwrap_or(0.0) as u64,
        })
    }

    fn gsum(&self) -> f64 {
        self.pins.iter().map(|p| 1.0 / p.1).sum()
    }

    /// Level 0..1 of full scale for the pin states in `word`.
    pub fn norm_word(&self, word: u64) -> f64 {
        let on: f64 = self.pins.iter().filter(|p| (word >> p.0) & 1 == 1).fold(0.0, |a, p| a + 1.0 / p.1);
        on / self.gsum()
    }

    /// Node voltage for a normalized level, with the load in the divider.
    pub fn volts(&self, norm: f64) -> f64 {
        let g = self.gsum();
        self.vdd * norm * g / (g + 1.0 / self.load_ohm)
    }

    /// Normalized level for every code, pin i of the list being bit i, for a plot of the DAC.
    pub fn transfer(&self) -> Vec<f64> {
        if self.pins.len() > 8 {
            return Vec::new();
        }
        (0..1u64 << self.pins.len())
            .map(|code| {
                let word = self.pins.iter().enumerate().fold(0u64, |w, (i, p)| w | (((code >> i) & 1) << p.0));
                self.norm_word(word)
            })
            .collect()
    }

    pub fn to_json(&self) -> String {
        let pins: Vec<String> = self.pins.iter().map(|p| format!("{{\"pin\":{},\"ohm\":{}}}", p.0, p.1)).collect();
        let tr: Vec<String> = self.transfer().iter().map(|v| format!("{v:.5}")).collect();
        let full = self.volts(1.0);
        format!(
            "{{\"pins\":[{}],\"load_ohm\":{},\"vdd\":{},\"window_cycles\":{},\"full_scale_v\":{full:.4},\"transfer\":[{}]}}",
            pins.join(","), self.load_ohm, self.vdd, self.window_cycles, tr.join(",")
        )
    }
}

// ------------------------------------------------------------------- input

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// Analog r, g and b channels with separate hsync and vsync pins.
    RgbSync,
    /// One analog channel that carries sync in its low levels.
    Composite,
}

#[derive(Clone, Debug)]
pub struct Input {
    pub name: String,
    pub kind: Kind,
    pub timing: Timing,
    pub hsync: Option<u8>,
    pub vsync: Option<u8>,
    pub sync_active_low: bool,
    pub channels: Vec<Channel>,
    /// Cycles per emulator quantum while capturing, 0 for the input's own choice.
    pub quantum: u32,
}

impl Input {
    pub fn parse(j: &J) -> Result<Input, String> {
        let name = j.str("name").ok_or("an input needs a \"name\"")?.to_string();
        let kind = match j.str("kind") {
            Some("rgb_sync") => Kind::RgbSync,
            Some("composite") => Kind::Composite,
            other => return Err(format!("{name}: kind is rgb_sync or composite, not {other:?}")),
        };
        let timing = Timing::parse(j.get("timing").ok_or(format!("{name}: no \"timing\""))?)?;
        let pin_of = |k: &str| -> Result<Option<u8>, String> {
            match j.get("pins").and_then(|p| p.num(k)) {
                Some(n) if (0.0..48.0).contains(&n) => Ok(Some(n as u8)),
                Some(_) => Err(format!("{name}: pin {k} is 0..47")),
                None => Ok(None),
            }
        };
        let (hsync, vsync) = (pin_of("hsync")?, pin_of("vsync")?);
        let mut channels = Vec::new();
        if let Some(J::Obj(list)) = j.get("channels") {
            for (cn, cj) in list {
                channels.push(Channel::parse(cn, cj)?);
            }
        }
        let has = |n: &str| channels.iter().any(|c| c.name == n);
        match kind {
            Kind::RgbSync => {
                if !(has("r") && has("g") && has("b")) {
                    return Err(format!("{name}: rgb_sync needs channels r, g and b"));
                }
                if hsync.is_none() || vsync.is_none() {
                    return Err(format!("{name}: rgb_sync needs pins hsync and vsync"));
                }
                if timing.pixel_hz <= 0.0 || timing.line_samples == 0 || timing.total_lines == 0 {
                    return Err(format!("{name}: timing needs pixel_hz, line_samples and total_lines"));
                }
            }
            Kind::Composite => {
                if !has("composite") {
                    return Err(format!("{name}: composite needs a channel named composite"));
                }
                if timing.line_samples == 0 || timing.total_lines == 0 {
                    return Err(format!("{name}: timing needs line_samples and total_lines"));
                }
            }
        }
        for c in channels.iter_mut() {
            if kind == Kind::Composite && c.window_cycles == 0 {
                c.window_cycles = timing.sample_cycles;
            }
        }
        if kind == Kind::Composite && channels.iter().any(|c| c.name == "composite" && c.window_cycles == 0) {
            return Err(format!("{name}: composite needs window_cycles or a timing with sample_cycles"));
        }
        Ok(Input { name, kind, timing, hsync, vsync, sync_active_low: !matches!(j.get("sync_active_low"), Some(J::Bool(false))), channels, quantum: j.num("quantum").unwrap_or(0.0) as u32 })
    }

    pub fn channel(&self, n: &str) -> &Channel {
        self.channels.iter().find(|c| c.name == n).expect("checked in parse")
    }

    /// Every pin the decoder reads, for the edge log.
    pub fn mask(&self) -> u64 {
        let mut m = 0u64;
        for c in &self.channels {
            for p in &c.pins {
                m |= 1u64 << p.0;
            }
        }
        for p in [self.hsync, self.vsync].into_iter().flatten() {
            m |= 1u64 << p;
        }
        m
    }

    /// Cycles per emulator quantum for a capture. Pin changes are exact at any quantum, but the
    /// DMA that feeds a PWM compare value runs once per quantum, so a PWM channel is only right
    /// while the quantum stays within one PWM period: the window of its channel. An RGB input
    /// is fed a word at a time by DMA, 8 pixels or more apart, and 64 cycles is well inside that.
    pub fn capture_quantum(&self) -> u32 {
        if self.quantum > 0 {
            return self.quantum;
        }
        match self.kind {
            Kind::RgbSync => 64,
            Kind::Composite => (self.channel("composite").window_cycles as u32).max(1),
        }
    }

    /// System cycles to capture so that one whole frame is inside: 2.2 frames.
    pub fn capture_cycles(&self, sys_hz: f64) -> u64 {
        let t = &self.timing;
        let line = match self.kind {
            Kind::RgbSync => t.line_samples as f64 * sys_hz / t.pixel_hz,
            Kind::Composite => (t.line_samples as u64 * t.sample_cycles) as f64,
        };
        (line * t.total_lines as f64 * 2.2) as u64
    }

    pub fn to_json(&self) -> String {
        let kind = if self.kind == Kind::RgbSync { "rgb_sync" } else { "composite" };
        let pin = |p: Option<u8>| p.map_or("null".to_string(), |n| n.to_string());
        let ch: Vec<String> = self.channels.iter().map(|c| format!("{}:{}", json_str(&c.name), c.to_json())).collect();
        format!(
            "{{\"name\":{},\"kind\":\"{kind}\",\"timing\":{},\"pins\":{{\"hsync\":{},\"vsync\":{}}},\"sync_active_low\":{},\"quantum\":{},\"capture_quantum\":{},\"channels\":{{{}}}}}",
            json_str(&self.name), self.timing.to_json(), pin(self.hsync), pin(self.vsync), self.sync_active_low, self.quantum, self.capture_quantum(), ch.join(",")
        )
    }
}

/// Inputs that ship with the server, as the JSON a client would send to video.set.
pub const PRESETS: &[(&str, &str)] = &[
    (
        "vga222_gpio6",
        "{\"name\":\"vga\",\"kind\":\"rgb_sync\",\"timing\":\"vga_640x480_60\",\"sync_active_low\":true,\
          \"pins\":{\"hsync\":12,\"vsync\":13},\"channels\":{\
          \"r\":{\"pins\":[{\"pin\":10,\"ohm\":1000},{\"pin\":11,\"ohm\":500}]},\
          \"g\":{\"pins\":[{\"pin\":8,\"ohm\":1000},{\"pin\":9,\"ohm\":500}]},\
          \"b\":{\"pins\":[{\"pin\":6,\"ohm\":1000},{\"pin\":7,\"ohm\":500}]}}}",
    ),
    (
        "ntsc_pwm28",
        "{\"name\":\"ntsc\",\"kind\":\"composite\",\"timing\":\"ntsc_m_22\",\
          \"channels\":{\"composite\":{\"pins\":[{\"pin\":28,\"ohm\":1000}],\"window_cycles\":22}}}",
    ),
];

// ------------------------------------------------------------------- edges

/// Pin changes `(cycle, pins)` from the emulator's edge log over `start..end`, end excluded.
/// The state at cycle t is that of the last entry at or before t.
pub struct EdgeTrack<'a> {
    pub e: &'a [(u64, u64)],
    pub start: u64,
    pub end: u64,
}

impl EdgeTrack<'_> {
    pub fn at(&self, t: u64) -> u64 {
        let i = self.e.partition_point(|x| x.0 <= t);
        if i == 0 { 0 } else { self.e[i - 1].1 }
    }

    /// Cycles in `t0..t1` during which `pin` is high.
    pub fn high_cycles(&self, pin: u8, t0: u64, t1: u64) -> u64 {
        let bit = |w: u64| (w >> pin) & 1 == 1;
        let mut j = self.e.partition_point(|x| x.0 <= t0);
        let mut cur = if j == 0 { false } else { bit(self.e[j - 1].1) };
        let (mut t, mut high) = (t0, 0u64);
        while j < self.e.len() && self.e[j].0 < t1 {
            if cur {
                high += self.e[j].0 - t;
            }
            t = self.e[j].0;
            cur = bit(self.e[j].1);
            j += 1;
        }
        if cur {
            high += t1 - t;
        }
        high
    }

    /// Times at which `pin` goes from low to high, after `start`.
    pub fn rises(&self, pin: u8) -> Vec<u64> {
        let mut out = Vec::new();
        let mut prev = false;
        for (k, &(t, w)) in self.e.iter().enumerate() {
            let now = (w >> pin) & 1 == 1;
            if k > 0 && now && !prev && t > self.start {
                out.push(t);
            }
            prev = now;
        }
        out
    }
}

// ----------------------------------------------------------------- decoder

pub struct RgbFrame {
    pub vsync_time: u64,
    pub lines: usize,
    pub vsync_lines: usize,
    /// line_samples x total_lines RGB888 from the frame's first hsync, blanking included.
    pub raster: Vec<u8>,
}

fn q8(norm: f64) -> u8 {
    (norm * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Rebuilds RGB frames of an rgb_sync input: one frame between two vsyncs.
pub fn decode_rgb(inp: &Input, tr: &EdgeTrack, sys_hz: f64) -> Vec<RgbFrame> {
    let t = &inp.timing;
    let (hp, vp) = (inp.hsync.unwrap(), inp.vsync.unwrap());
    let act = |w: u64, p: u8| ((w >> p) & 1 == 0) == inp.sync_active_low;
    let mut starts: Vec<u64> = Vec::new();
    let mut prev = false;
    for (k, &(time, w)) in tr.e.iter().enumerate() {
        let now = act(w, hp);
        if k > 0 && now && !prev && time > tr.start {
            starts.push(time);
        }
        prev = now;
    }
    let sample_cycles = sys_hz / t.pixel_hz;
    let vs_at = |i: usize| act(tr.at(starts[i] + 4), vp);
    let mut vstarts = Vec::new();
    for k in 1..starts.len() {
        if vs_at(k) && !vs_at(k - 1) {
            vstarts.push(k);
        }
    }
    let (r, g, b) = (inp.channel("r"), inp.channel("g"), inp.channel("b"));
    let mut frames = Vec::new();
    for w in vstarts.windows(2) {
        let (k0, k1) = (w[0], w[1]);
        let mut raster = vec![0u8; t.line_samples * t.total_lines * 3];
        let mut vsync_lines = 0;
        for (l, k) in (k0..k1).enumerate() {
            if vs_at(k) {
                vsync_lines += 1;
            }
            if l >= t.total_lines {
                continue;
            }
            for x in 0..t.line_samples {
                let at = starts[k] + ((x as f64 + 0.5) * sample_cycles) as u64;
                let word = if at < tr.end { tr.at(at) } else { 0 };
                let o = (l * t.line_samples + x) * 3;
                raster[o] = q8(r.norm_word(word));
                raster[o + 1] = q8(g.norm_word(word));
                raster[o + 2] = q8(b.norm_word(word));
            }
        }
        frames.push(RgbFrame { vsync_time: starts[k0], lines: k1 - k0, vsync_lines, raster });
    }
    frames
}

pub struct CompositeReport {
    /// Window levels 0..1, one per sample.
    pub levels: Vec<f64>,
    pub lines: usize,
    pub hsync_samples: (usize, usize),
    pub frame_lines: Vec<usize>,
    /// One frame of line_samples x total_lines levels, from the first vsync.
    pub frame: Option<Vec<f64>>,
}

/// Windows the composite channel, finds sync by level and cuts out a frame.
pub fn decode_composite(inp: &Input, tr: &EdgeTrack) -> CompositeReport {
    let t = &inp.timing;
    let ch = inp.channel("composite");
    let w = ch.window_cycles;
    let mut rep = CompositeReport { levels: Vec::new(), lines: 0, hsync_samples: (0, 0), frame_lines: Vec::new(), frame: None };
    let mut rises: Vec<u64> = Vec::new();
    for c in &ch.pins {
        rises.extend(tr.rises(c.0));
    }
    rises.sort_unstable();
    if rises.len() < 100 {
        return rep;
    }
    let mut at = tr.start + (rises[0] - tr.start) % w;
    let gsum: f64 = ch.pins.iter().map(|p| 1.0 / p.1).sum();
    while at + w <= tr.end {
        let on = ch.pins.iter().fold(0.0, |a, p| a + tr.high_cycles(p.0, at, at + w) as f64 / p.1);
        rep.levels.push(on / gsum / w as f64);
        at += w;
    }
    let low = |l: f64| l < t.sync_thr;
    let mut runs = Vec::new();
    let mut s = None;
    for (k, &l) in rep.levels.iter().enumerate() {
        match (low(l), s) {
            (true, None) => s = Some(k),
            (false, Some(st)) => {
                runs.push((st, k - st));
                s = None;
            }
            _ => {}
        }
    }
    let hs: Vec<&(usize, usize)> = runs.iter().filter(|r| r.1 >= t.hs_min && r.1 <= t.hs_max).collect();
    rep.lines = hs.len();
    if hs.len() > 2 {
        rep.hsync_samples = (hs.iter().map(|r| r.1).min().unwrap(), hs.iter().map(|r| r.1).max().unwrap());
    }
    let vs: Vec<usize> = runs.iter().filter(|r| r.1 > t.line_samples / 2).map(|r| r.0).collect();
    let mut vfirst = Vec::new();
    for (j, &v) in vs.iter().enumerate() {
        if j == 0 || v - vs[j - 1] > t.line_samples * 20 {
            vfirst.push(v);
        }
    }
    for p in vfirst.windows(2) {
        rep.frame_lines.push(((p[1] - p[0]) as f64 / t.line_samples as f64).round() as usize);
    }
    if let Some(&v0) = vfirst.first() {
        let n = t.line_samples * t.total_lines;
        if v0 + n <= rep.levels.len() {
            rep.frame = Some(rep.levels[v0..v0 + n].to_vec());
        }
    }
    rep
}

/// Crops the active area of an RGB raster.
pub fn crop_active(t: &Timing, raster: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(t.active_w * t.active_lines * 3);
    for l in t.vsync_to_active..t.vsync_to_active + t.active_lines {
        let o = (l * t.line_samples + t.active_offset) * 3;
        out.extend_from_slice(&raster[o..o + t.active_w * 3]);
    }
    out
}

/// A composite frame as gray RGB, four samples averaged to take the colour burst out.
pub fn composite_gray(t: &Timing, levels: &[f64]) -> Vec<u8> {
    let (w, h) = (t.line_samples, levels.len() / t.line_samples);
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let avg: f64 = (0..4).map(|k| levels[y * w + (x + k).min(w - 1)]).sum::<f64>() / 4.0;
            let v = q8(avg);
            out.extend_from_slice(&[v, v, v]);
        }
    }
    out
}
