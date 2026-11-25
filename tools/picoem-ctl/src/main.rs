//! picoem-ctl: boots an RP2350 firmware in picoem and serves the 86Box remote machine
//! monitor dialect on a TCP port, so the Python Rmm client, FramePoller and the xp5-studio
//! manager drive it like any other machine.
//!
//! Wire format: a banner line "OK picoem-ctl 0.1 paused" or "... running", then one JSON-RPC
//! 2.0 object per line. One client at a time, 127.0.0.1 only.
//!
//! Implemented: rpc.hello, machine.status|pause|resume|quit, cpu.regs.get, mem.phys.read,
//! capture.screenshot, video.get|set|presets, picoem.info and picoem.gpio. Anything else
//! answers -32601.
//!
//! Video inputs are data, see video.rs: pin roles, resistor networks per analog channel and a
//! timing. A screenshot logs the pin edges for 2.2 frames of guest time at the input's capture quantum, so it
//! advances the guest, and decodes the newest whole frame of the named input. A paused
//! screenshot returns the newest frame already taken. `--verify-video` decodes one capture with
//! the per-cycle reference decoder of picoem_ntsc_sim and with the edge decoder and compares.
//! `--verify-quantum Q` compares a capture at quantum 1 with one at Q. `--trace FROM TO Q` prints
//! which function each core runs between two cycles counted from the end of the boot, or with
//! TRACE_STEP=N the cores' counters every N cycles.

#![allow(dead_code)]

use rp2350_emu::{Config, Emulator};
use std::collections::HashMap;
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::exit;
use std::time::Duration;

mod json;
mod video;

use json::*;

include!("sim.rs");

// ------------------------------------------------------------------ server

struct Args {
    elf: String,
    bin: String,
    rom: String,
    port: u16,
    boot_limit_ms: u64,
    verify_video: bool,
    verify_quantum: u32,
    trace: Option<(u64, u64, u32)>,
}

fn parse_args() -> Args {
    let dir = env!("CARGO_MANIFEST_DIR");
    let mut a = Args {
        elf: String::new(),
        bin: String::new(),
        rom: format!("{dir}/../../GIT_SRC_DIR/picoem/roms/rp2350/bootrom-combined.bin"),
        port: 0,
        boot_limit_ms: 1500,
        verify_video: false,
        verify_quantum: 0,
        trace: None,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let flag = v[i].as_str();
        let mut val = || {
            i += 1;
            v.get(i).cloned().unwrap_or_else(|| fail(&format!("{flag} needs a value")))
        };
        match flag {
            "--elf" => a.elf = val(),
            "--bin" => a.bin = val(),
            "--rom" => a.rom = val(),
            "--port" => a.port = val().parse().unwrap_or_else(|_| fail("--port")),
            "--verify-video" => a.verify_video = true,
            "--trace" => {
                let (s0, e0, q) = (val(), val(), val());
                a.trace = Some((s0.parse().unwrap_or_else(|_| fail("--trace")), e0.parse().unwrap_or_else(|_| fail("--trace")), q.parse().unwrap_or_else(|_| fail("--trace"))));
            }
            "--verify-quantum" => a.verify_quantum = val().parse().unwrap_or_else(|_| fail("--verify-quantum")),
            "--boot-limit-ms" => a.boot_limit_ms = val().parse().unwrap_or_else(|_| fail("--boot-limit-ms")),
            "-h" | "--help" => {
                println!("usage: picoem-ctl --elf F --port N [--bin F] [--rom F] [--boot-limit-ms N] [--verify-video] [--verify-quantum Q] [--trace FROM TO Q]");
                exit(0);
            }
            _ => fail(&format!("unknown argument {flag}")),
        }
        i += 1;
    }
    if a.elf.is_empty() || (a.port == 0 && !a.verify_video && a.verify_quantum == 0 && a.trace.is_none()) {
        fail("--elf and --port are required");
    }
    if a.bin.is_empty() {
        a.bin = a.elf.trim_end_matches(".elf").to_string() + ".bin";
    }
    a
}

struct Frame {
    w: usize,
    h: usize,
    seq: u64,
    rgb: Vec<u8>,
}

struct Server {
    sim: Sim,
    running: bool,
    quit: bool,
    inputs: Vec<video::Input>,
    frames: HashMap<String, Frame>,
    seq: u64,
}

fn ok(id: &J, result: &str) -> String {
    format!("{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{result}}}\n", id_text(id))
}

fn err(id: &J, code: i32, msg: &str) -> String {
    format!("{{\"jsonrpc\":\"2.0\",\"id\":{},\"error\":{{\"code\":{code},\"message\":{}}}}}\n", id_text(id), json_str(msg))
}

fn id_text(id: &J) -> String {
    match id {
        J::Num(n) => format!("{}", *n as i64),
        J::Str(s) => json_str(s),
        _ => "null".into(),
    }
}

fn event(name: &str, params: &str) -> String {
    format!("{{\"jsonrpc\":\"2.0\",\"method\":\"{name}\",\"params\":{params}}}\n")
}

impl Server {
    fn state(&self) -> &'static str {
        if self.running { "running" } else { "paused" }
    }

    /// Runs 2.2 frames of guest time at the input's capture quantum with the pin edges logged, then decodes the
    /// newest whole frame of input `idx`.
    fn grab_frame(&mut self, idx: usize) -> Result<(), String> {
        let inp = self.inputs[idx].clone();
        let sys = self.sim.emu.bus.sys_clk_hz() as f64;
        self.sim.emu.gpio_edge_log_enable(inp.mask(), 8_000_000);
        let start = self.sim.emu.cycles() + 1;
        self.sim.run_for(inp.capture_cycles(sys), inp.capture_quantum(), |_| false);
        let end = self.sim.emu.cycles() + 1;
        let edges = self.sim.emu.gpio_edge_log_take();
        let dropped = self.sim.emu.gpio_edge_log_dropped();
        self.sim.emu.gpio_edge_log_disable();
        if dropped > 0 {
            return Err(format!("{}: edge log overflowed by {dropped} entries", inp.name));
        }
        let tr = video::EdgeTrack { e: &edges, start, end };
        let t = &inp.timing;
        let (w, h, rgb) = match inp.kind {
            video::Kind::RgbSync => {
                let frames = video::decode_rgb(&inp, &tr, sys);
                let f = frames.last().ok_or(format!("{}: no complete frame in the capture", inp.name))?;
                (t.active_w, t.active_lines, video::crop_active(t, &f.raster))
            }
            video::Kind::Composite => {
                let rep = video::decode_composite(&inp, &tr);
                let f = rep.frame.ok_or(format!("{}: no complete frame in the capture", inp.name))?;
                (t.line_samples, t.total_lines, video::composite_gray(t, &f))
            }
        };
        self.seq += 1;
        self.frames.insert(inp.name.clone(), Frame { w, h, seq: self.seq, rgb });
        Ok(())
    }

    fn set_inputs(&mut self, list: Vec<video::Input>) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        for i in &list {
            if !seen.insert(i.name.clone()) {
                return Err(format!("two inputs are named {}", i.name));
            }
        }
        self.inputs = list;
        self.frames.clear();
        Ok(())
    }

    fn inputs_json(&self) -> String {
        let v: Vec<String> = self.inputs.iter().map(|i| i.to_json()).collect();
        format!("{{\"inputs\":[{}]}}", v.join(","))
    }

    fn handle(&mut self, line: &str, out: &mut Vec<String>) {
        let req = match parse_json(line) {
            Ok(r) => r,
            Err(e) => return out.push(err(&J::Null, -32700, &format!("parse error: {e}"))),
        };
        let id = req.get("id").cloned().unwrap_or(J::Null);
        let method = match req.str("method") {
            Some(m) => m.to_string(),
            None => return out.push(err(&id, -32600, "no method")),
        };
        let empty = J::Obj(Vec::new());
        let p = req.get("params").unwrap_or(&empty);
        match method.as_str() {
            "rpc.hello" => out.push(ok(&id, "{\"protocol\":1,\"server\":\"picoem-ctl\",\"version\":\"0.1\"}")),
            "machine.status" => {
                let r = format!("{{\"state\":\"{}\",\"tsc\":{}}}", self.state(), self.sim.emu.cycles());
                out.push(ok(&id, &r));
            }
            "machine.pause" => {
                if self.running {
                    self.running = false;
                    out.push(ok(&id, "{}"));
                    out.push(event("machine.stopped", "{\"reason\":\"pause\"}"));
                } else {
                    out.push(ok(&id, "{}"));
                }
            }
            "machine.resume" => {
                if !self.running {
                    self.running = true;
                    out.push(ok(&id, "{}"));
                    out.push(event("machine.resumed", "{}"));
                } else {
                    out.push(ok(&id, "{}"));
                }
            }
            "machine.quit" => {
                self.quit = true;
                out.push(ok(&id, "{}"));
            }
            "cpu.regs.get" => {
                if self.running {
                    return out.push(err(&id, 1, "not paused"));
                }
                let core = (p.num("core").unwrap_or(0.0) as usize).min(1);
                let c = self.sim.emu.core(core);
                let mut parts: Vec<String> = (0..13).map(|i| format!("\"r{i}\":{}", c.regs.r[i])).collect();
                parts.push(format!("\"sp\":{}", c.regs.r[13]));
                parts.push(format!("\"lr\":{}", c.regs.r[14]));
                parts.push(format!("\"pc\":{}", c.regs.pc()));
                parts.push(format!("\"xpsr\":{}", c.regs.xpsr));
                parts.push(format!("\"core\":{core}"));
                out.push(ok(&id, &format!("{{{}}}", parts.join(","))));
            }
            "mem.phys.read" => {
                if self.running {
                    return out.push(err(&id, 1, "not paused"));
                }
                let (addr, len) = (p.num("addr").unwrap_or(-1.0), p.num("len").unwrap_or(-1.0));
                if addr < 0.0 || len <= 0.0 || len > 65536.0 || addr + len > 4294967296.0 {
                    return out.push(err(&id, -32602, "addr and len out of range"));
                }
                let (addr, len) = (addr as u32, len as u32);
                let mut hex = String::with_capacity(len as usize * 2);
                let (first, last) = (addr & !3, (addr + len - 1) & !3);
                let mut a = first;
                loop {
                    let w = self.sim.emu.peek(a).to_le_bytes();
                    for (k, b) in w.iter().enumerate() {
                        let at = a + k as u32;
                        if at >= addr && at < addr + len {
                            hex.push_str(&format!("{b:02x}"));
                        }
                    }
                    if a == last {
                        break;
                    }
                    a += 4;
                }
                out.push(ok(&id, &format!("{{\"data\":\"{hex}\"}}")));
            }
            "capture.screenshot" => {
                let idx = match p.str("input") {
                    Some(n) => self.inputs.iter().position(|i| i.name == n),
                    None => (!self.inputs.is_empty()).then_some(0),
                };
                let Some(idx) = idx else {
                    return out.push(err(&id, -32602, "no such video input"));
                };
                if self.running {
                    if let Err(e) = self.grab_frame(idx) {
                        return out.push(err(&id, 8, &e));
                    }
                }
                match self.frames.get(&self.inputs[idx].name) {
                    None => out.push(err(&id, 4, "no frame yet; resume the machine for one")),
                    Some(f) => {
                        let fresh = self.running;
                        let r = format!(
                            "{{\"w\":{},\"h\":{},\"src_w\":{},\"src_h\":{},\"seq\":{},\"fresh\":{fresh},\"input\":{},\"data\":\"{}\"}}",
                            f.w, f.h, f.w, f.h, f.seq, json_str(&self.inputs[idx].name), base64(&f.rgb)
                        );
                        out.push(ok(&id, &r));
                    }
                }
            }
            "video.get" => out.push(ok(&id, &self.inputs_json())),
            "video.presets" => {
                let pr: Vec<String> = video::PRESETS
                    .iter()
                    .map(|(n, spec)| format!("{{\"name\":{},\"spec\":{spec}}}", json_str(n)))
                    .collect();
                let tm: Vec<String> = video::TIMING_NAMES.iter().filter_map(|n| video::Timing::preset(n)).map(|t| t.to_json()).collect();
                out.push(ok(&id, &format!("{{\"presets\":[{}],\"timings\":[{}]}}", pr.join(","), tm.join(","))));
            }
            "video.set" => {
                let mut list = Vec::new();
                let parsed: Result<(), String> = (|| {
                    if let Some(J::Arr(a)) = p.get("inputs") {
                        for j in a {
                            list.push(video::Input::parse(j)?);
                        }
                    } else if let Some(n) = p.str("preset") {
                        let spec = video::PRESETS.iter().find(|x| x.0 == n).ok_or(format!("unknown preset {n}"))?;
                        list.push(video::Input::parse(&parse_json(spec.1)?)?);
                    } else {
                        return Err("video.set takes \"inputs\": [...] or \"preset\": name".into());
                    }
                    self.set_inputs(std::mem::take(&mut list))
                })();
                match parsed {
                    Ok(()) => out.push(ok(&id, &self.inputs_json())),
                    Err(e) => out.push(err(&id, -32602, &e)),
                }
            }
            "picoem.info" => {
                let e = &self.sim.emu;
                let r = format!(
                    "{{\"sys_hz\":{},\"cycles\":{},\"core1_running\":{},\"pc0\":{},\"pc1\":{}}}",
                    e.bus.sys_clk_hz(), e.cycles(), self.sim.core1_running, e.core(0).regs.pc(), e.core(1).regs.pc()
                );
                out.push(ok(&id, &r));
            }
            "picoem.gpio" => out.push(ok(&id, &format!("{{\"pins\":{}}}", self.sim.emu.gpio_read_all()))),
            other => out.push(err(&id, -32601, &format!("{other}: not available on picoem-ctl"))),
        }
    }

    /// Serves one connected client until it leaves or quits. The emulator free-runs between
    /// requests while running.
    fn serve(&mut self, mut s: TcpStream) -> std::io::Result<()> {
        s.set_nodelay(true)?;
        s.set_nonblocking(true)?;
        s.write_all(format!("OK picoem-ctl 0.1 {}\n", self.state()).as_bytes())?;
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            let mut got = false;
            match s.read(&mut chunk) {
                Ok(0) => return Ok(()),
                Ok(n) => {
                    buf.extend_from_slice(&chunk[..n]);
                    got = true;
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => {}
                Err(e) => return Err(e),
            }
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                let text = String::from_utf8_lossy(&line).trim().to_string();
                if text.is_empty() {
                    continue;
                }
                let mut out = Vec::new();
                self.handle(&text, &mut out);
                s.set_nonblocking(false)?;
                for o in out {
                    s.write_all(o.as_bytes())?;
                }
                s.set_nonblocking(true)?;
                if self.quit {
                    return Ok(());
                }
            }
            if self.running {
                self.sim.run_for(1_000_000, 1024, |_| false);
            } else if !got {
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    }
}

fn boot_sim(args: &Args) -> Sim {
    let elf = fs::read(&args.elf).unwrap_or_else(|e| fail(&format!("{}: {e}", args.elf)));
    let fw = fs::read(&args.bin).unwrap_or_else(|e| fail(&format!("{}: {e}", args.bin)));
    let rom = fs::read(&args.rom).unwrap_or_else(|e| fail(&format!("{}: {e}", args.rom)));
    let syms = read_symbols(&elf).unwrap_or_else(|e| fail(&e));
    let mut sim = build_sim(&fw, &rom, &syms);

    let vga_pins = 0xffu32 << VGA_BASE_PIN;
    sim.run_for(args.boot_limit_ms * (SYS_HZ / 1000), 256, |s| {
        s.core1_running && (s.emu.bus.pio[0].pad_oe & vga_pins) == vga_pins
    });
    if !sim.core1_running || (sim.emu.bus.pio[0].pad_oe & vga_pins) != vga_pins {
        fail("firmware never started driving the VGA pins");
    }
    let sys_hz = sim.emu.bus.sys_clk_hz() as u64;
    sim.run_for(sys_hz / 20, 256, |_| false);
    eprintln!("picoem-ctl: booted at cycle {}, sys clock {} Hz", sim.emu.cycles(), sys_hz);
    sim
}

fn main() {
    let args = parse_args();
    let sim = boot_sim(&args);
    let mut sim = sim;
    if let Some((from, to, q)) = args.trace {
        trace_window(&args, &mut sim, from, to, q);
        exit(0);
    }
    if args.verify_quantum > 0 {
        let b = boot_sim(&args);
        exit(if verify_quantum(&mut sim, b, args.verify_quantum) { 0 } else { 1 });
    }

    if args.verify_video {
        exit(if verify_video(&mut sim) { 0 } else { 1 });
    }

    let listener = TcpListener::bind(("127.0.0.1", args.port)).unwrap_or_else(|e| fail(&format!("port {}: {e}", args.port)));
    println!("picoem-ctl: listening on 127.0.0.1:{}", args.port);
    let _ = std::io::stdout().flush();
    let mut srv = Server { sim, running: false, quit: false, inputs: Vec::new(), frames: HashMap::new(), seq: 0 };
    let defaults: Vec<video::Input> = video::PRESETS
        .iter()
        .map(|(_, spec)| video::Input::parse(&parse_json(spec).expect("preset json")).expect("preset input"))
        .collect();
    srv.set_inputs(defaults).expect("preset names differ");
    while !srv.quit {
        match listener.accept() {
            Ok((s, _)) => {
                if let Err(e) = srv.serve(s) {
                    eprintln!("picoem-ctl: client dropped: {e}");
                }
            }
            Err(e) => {
                eprintln!("picoem-ctl: accept: {e}");
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

/// Decodes one capture twice, with the per-cycle sampler and decoder of picoem_ntsc_sim and with
/// the edge log and the data-driven decoder, and says whether the frames agree.
fn verify_video(sim: &mut Sim) -> bool {
    let sys = sim.emu.bus.sys_clk_hz() as f64;
    let inputs: Vec<video::Input> = video::PRESETS
        .iter()
        .map(|(_, spec)| video::Input::parse(&parse_json(spec).expect("preset json")).expect("preset input"))
        .collect();
    let mask = inputs.iter().fold(0u64, |m, i| m | i.mask());
    sim.emu.gpio_edge_log_enable(mask, 8_000_000);
    let (t0, vga, ntsc) = capture(sim, (sys / 60.0 * 2.2) as u64);
    let edges = sim.emu.gpio_edge_log_take();
    sim.emu.gpio_edge_log_disable();
    println!("verify: {} cycles, {} edges, {} bytes per frame sampled", vga.len(), edges.len(), vga.len() * 2);
    let tr = video::EdgeTrack { e: &edges, start: t0, end: t0 + vga.len() as u64 };
    let mut good = true;

    let old = decode_vga(&vga, t0, sys);
    let new = video::decode_rgb(&inputs[0], &tr, sys);
    println!("vga: old {} frames, new {} frames", old.frames.len(), new.len());
    good &= !old.frames.is_empty() && old.frames.len() == new.len();
    for (k, (a, b)) in old.frames.iter().zip(new.iter()).enumerate() {
        let same = a.raster == b.raster && a.lines == b.lines && a.vsync_lines == b.vsync_lines && a.vsync_cycle == b.vsync_time;
        println!("vga frame {k}: raster fnv old {:016x} new {:016x}, {}", fnv(a.raster.iter().copied()), fnv(b.raster.iter().copied()), if same { "same" } else { "DIFFERENT" });
        good &= same;
    }

    let old = decode_ntsc(&ntsc);
    let new = video::decode_composite(&inputs[1], &tr);
    let new_levels: Vec<u8> = new.levels.iter().map(|l| ((l * 22.0).round() as u32 / 2) as u8).collect();
    let old_frame_same = match (&old.luma, &new.frame) {
        (Some(a), Some(b)) => {
            let nb: Vec<u8> = b.iter().map(|l| ((l * 22.0).round() as u32 / 2) as u8).collect();
            &nb == a
        }
        (None, None) => true,
        _ => false,
    };
    println!("ntsc: old {} lines, new {} lines, {} window levels, frame lines old {:?} new {:?}", old.lines, new.lines, new_levels.len(), old.frame_lines, new.frame_lines);
    if let (Some(a), Some(_)) = (&old.luma, &new.frame) {
        println!("ntsc frame: levels fnv old {:016x}, {}", fnv(a.iter().copied()), if old_frame_same { "same" } else { "DIFFERENT" });
    }
    good &= old.lines == new.lines && old.frame_lines == new.frame_lines && old_frame_same && old.luma.is_some();
    println!("RESULT: {}", if good { "PASS" } else { "FAIL" });
    good
}

/// One logged capture of `cycles` at quantum `q`: the edges, and the host seconds it took.
fn logged_capture(sim: &mut Sim, mask: u64, cycles: u64, q: u32) -> (Vec<(u64, u64)>, u64, u64, f64) {
    sim.emu.gpio_edge_log_enable(mask, 16_000_000);
    let start = sim.emu.cycles() + 1;
    let t = std::time::Instant::now();
    sim.run_for(cycles, q, |_| false);
    let secs = t.elapsed().as_secs_f64();
    let end = sim.emu.cycles() + 1;
    let edges = sim.emu.gpio_edge_log_take();
    sim.emu.gpio_edge_log_disable();
    (edges, start, end, secs)
}

/// Keeps the entries at which the masked pins change.
fn masked_changes(e: &[(u64, u64)], mask: u64) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    for &(t, w) in e {
        let m = w & mask;
        if out.last().map(|l| l.1) != Some(m) {
            out.push((t, m));
        }
    }
    out
}

/// Captures the same guest time at quantum 1 and at quantum `q` on two identically booted
/// emulators and compares the pin edges and the decoded frames.
fn verify_quantum(a: &mut Sim, mut b: Sim, q: u32) -> bool {
    let sys = a.emu.bus.sys_clk_hz() as f64;
    let inputs: Vec<video::Input> = video::PRESETS
        .iter()
        .map(|(_, spec)| video::Input::parse(&parse_json(spec).expect("preset json")).expect("preset input"))
        .collect();
    let mask = inputs.iter().fold(0u64, |m, i| m | i.mask());
    let cycles = ((sys / 60.0 * 2.2) as u64).div_ceil(q as u64) * q as u64;
    let (ea, sa, na, ta) = logged_capture(a, mask, cycles, 1);
    let (eb, sb, nb, tb) = logged_capture(&mut b, mask, cycles, q);
    println!("q1: {} entries in {ta:.1} s, {:.2} Mcycles/s; q{q}: {} entries in {tb:.1} s, {:.2} Mcycles/s, window start {} vs {}", ea.len(), cycles as f64 / ta / 1e6, eb.len(), cycles as f64 / tb / 1e6, sa, sb);
    let mut good = sa == sb && na == nb;
    for inp in &inputs {
        let (ca, cb) = (masked_changes(&ea, inp.mask()), masked_changes(&eb, inp.mask()));
        let diff = ca.iter().zip(cb.iter()).filter(|(x, y)| x != y).count() + ca.len().abs_diff(cb.len());
        let first = ca.iter().zip(cb.iter()).position(|(x, y)| x != y);
        if let Some(k) = first {
            for j in k.saturating_sub(8)..(k + 10).min(ca.len().min(cb.len())) {
                println!("  change {j}: q1 {:>10} {:#06x} (+{:>5})   q{q} {:>10} {:#06x} (+{:>5})", ca[j].0, ca[j].1, ca[j].0 - ca.get(j.wrapping_sub(1)).map_or(0, |x| x.0), cb[j].0, cb[j].1, cb[j].0 - cb.get(j.wrapping_sub(1)).map_or(0, |x| x.0));
            }
        }
        println!("{}: q1 {} changes, q{q} {} changes, {diff} differ{}", inp.name, ca.len(), cb.len(),
                 first.map_or(String::new(), |k| format!(", first at change {k}: q1 {:?} q{q} {:?}", ca[k], cb[k])));
        let (ta_, tb_) = (video::EdgeTrack { e: &ea, start: sa, end: na }, video::EdgeTrack { e: &eb, start: sb, end: nb });
        let (ha, hb) = match inp.kind {
            video::Kind::RgbSync => {
                let f = |t: &video::EdgeTrack| video::decode_rgb(inp, t, sys).last().map(|f| fnv(f.raster.iter().copied()));
                (f(&ta_), f(&tb_))
            }
            video::Kind::Composite => {
                let f = |t: &video::EdgeTrack| video::decode_composite(inp, t).frame.map(|f| fnv(f.iter().map(|l| (l * 22.0).round() as u8)));
                (f(&ta_), f(&tb_))
            }
        };
        println!("{}: frame hash q1 {:?} q{q} {:?}", inp.name, ha, hb);
        if inp.kind == video::Kind::RgbSync {
            let (fa, fb) = (video::decode_rgb(inp, &ta_, sys), video::decode_rgb(inp, &tb_, sys));
            if let (Some(x), Some(y)) = (fa.last(), fb.last()) {
                let n = x.raster.len() / 3;
                let bad = (0..n).filter(|&i| x.raster[i * 3..i * 3 + 3] != y.raster[i * 3..i * 3 + 3]).count();
                let lines_bad = (0..inp.timing.total_lines)
                    .filter(|&l| x.raster[l * inp.timing.line_samples * 3..(l + 1) * inp.timing.line_samples * 3] != y.raster[l * inp.timing.line_samples * 3..(l + 1) * inp.timing.line_samples * 3])
                    .count();
                println!("{}: {bad} of {n} pixels differ, {lines_bad} of {} lines; frame vsync q1 {} q{q} {}", inp.name, inp.timing.total_lines, x.vsync_time, y.vsync_time);
            }
        }
        good &= diff == 0;
    }
    println!("RESULT: {}", if good { "PASS" } else { "FAIL" });
    good
}

fn sym_name(syms: &HashMap<String, Sym>, pc: u32) -> String {
    syms.iter()
        .filter(|(_, s)| pc >= s.addr && pc < s.addr + s.size.max(1))
        .map(|(n, _)| n.clone())
        .next()
        .unwrap_or_else(|| format!("{pc:#x}"))
}

/// Prints which function each core is in, as runs, from guest cycle `from` to `to`, stepping at
/// quantum `q` from `from` on. Cycles are counted from the end of the boot.
fn trace_window(args: &Args, sim: &mut Sim, from: u64, to: u64, q: u32) {
    let syms = read_symbols(&fs::read(&args.elf).expect("elf")).expect("symbols");
    let base = sim.emu.cycles();
    sim.run_for(from, 256, |_| false);
    if let Some(step) = std::env::var("TRACE_STEP").ok().and_then(|v| v.parse::<u64>().ok()) {
        for _ in 0..(to - from) / step {
            sim.run_for(step, q, |_| false);
            let (c0, c1) = (sim.emu.core(0).counters.clone(), sim.emu.core(1).counters.clone());
            println!("{:>9} core0 pc {:08x} instr {:>9} busy {:>9} idle {:>8} | core1 pc {:08x} instr {:>9} busy {:>9} idle {:>8}",
                     sim.emu.cycles() - base, sim.emu.core(0).regs.pc(), c0.instructions, c0.decode_execute_cycles, c0.idle_skipped_cycles,
                     sim.emu.core(1).regs.pc(), c1.instructions, c1.decode_execute_cycles, c1.idle_skipped_cycles);
        }
        return;
    }
    let mut runs: Vec<(u64, [String; 2])> = Vec::new();
    let t0 = sim.emu.cycles();
    sim.run_for(to - from, q, |s| {
        let now = [sym_name(&syms, s.emu.core(0).regs.pc()), sym_name(&syms, s.emu.core(1).regs.pc())];
        if runs.last().map(|r| &r.1) != Some(&now) {
            runs.push((s.emu.cycles() - base, now));
        }
        false
    });
    println!("trace q{q}: {} runs from {} (boot + {from}), start {t0}", runs.len(), base + from);
    for (t, n) in runs.iter() {
        println!("{t:>9} core0 {:<40} core1 {}", n[0], n[1]);
    }
}
