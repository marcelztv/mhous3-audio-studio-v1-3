"""Insert effects.  Every effect is registered with its parameters so the UI can build controls automatically.

fn(x, **params): x is float32 (n, 2), returns the same shape.  All effects are offline / vectorised or block-based."""
import numpy as np
from scipy import signal

from . import dsp
from .dsp import SR

EFFECTS = {}


def effect(eid, name, cat, params):
    """params: list of (key, label, lo, hi, default, step, unit)  or  (key, label, [choices], default)."""
    def deco(fn):
        EFFECTS[eid] = dict(id=eid, name=name, cat=cat, params=params, fn=fn)
        return fn
    return deco


def defaults(eid):
    return {p[0]: (p[3] if isinstance(p[2], list) else p[4]) for p in EFFECTS[eid]["params"]}


def categories():
    out = {}
    for e in EFFECTS.values():
        out.setdefault(e["cat"], []).append(e)
    return out


def _mix(dry, wet, m):
    return (dry * (1 - m) + wet * m).astype(np.float32)


def run(eid, x, params=None):
    p = defaults(eid)
    p.update(params or {})
    return np.asarray(EFFECTS[eid]["fn"](np.asarray(x, np.float32), **p), np.float32)


# ============================================================================ dynamics
@effect("comp", "Compressor", "Dynamics", [("thr", "Threshold", -60, 0, -20, 1, "dB"), ("ratio", "Ratio", 1, 20, 3, 0.1, ":1"),
        ("attack", "Attack", 0.5, 100, 10, 0.5, "ms"), ("release", "Release", 10, 1000, 120, 5, "ms"), ("knee", "Knee", 0, 24, 6, 1, "dB"),
        ("makeup", "Make-up", 0, 24, 0, 0.5, "dB"), ("mix", "Mix", 0, 1, 1, 0.01, "")])
def _comp(x, thr, ratio, attack, release, knee, makeup, mix):
    return dsp.compressor(x, SR, thr, ratio, attack, release, knee, makeup, mix)


@effect("limiter", "Limiter", "Dynamics", [("ceiling", "Ceiling", -12, 0, -1, 0.1, "dB"), ("look", "Look-ahead", 1, 20, 6, 1, "ms")])
def _limiter(x, ceiling, look):
    return dsp.limiter(x, SR, ceiling, look)


@effect("brick", "Brickwall Limiter", "Dynamics", [("ceiling", "Ceiling", -12, 0, -0.3, 0.1, "dB")])
def _brick(x, ceiling):
    return dsp.limiter(x, SR, ceiling, 5.0, true_peak=True)


@effect("maximizer", "Maximizer", "Dynamics", [("gain", "Gain", 0, 24, 6, 0.5, "dB"), ("ceiling", "Ceiling", -12, 0, -0.5, 0.1, "dB")])
def _maximizer(x, gain, ceiling):
    return dsp.limiter(x * float(dsp.db2lin(gain)), SR, ceiling, 8.0, true_peak=True)


@effect("deess", "De-esser", "Dynamics", [("freq", "Frequency", 3500, 10000, 6500, 100, "Hz"), ("thr", "Threshold", -60, -10, -34, 1, "dB"),
        ("ratio", "Ratio", 1, 10, 3, 0.5, ":1")])
def _deess(x, freq, thr, ratio):
    return dsp.deesser(x, SR, freq, thr, ratio)


@effect("gate", "Gate", "Dynamics", [("thr", "Threshold", -80, -10, -50, 1, "dB"), ("range", "Range", -80, -3, -40, 1, "dB"),
        ("attack", "Attack", 0.5, 50, 3, 0.5, "ms"), ("release", "Release", 10, 800, 150, 5, "ms")])
def _gate(x, thr, range, attack, release):
    return dsp.expander(x, SR, thr, 8.0, range, attack, release)


@effect("expander", "Expander", "Dynamics", [("thr", "Threshold", -80, -10, -45, 1, "dB"), ("ratio", "Ratio", 1, 6, 2, 0.1, ":1"),
        ("range", "Range", -40, -3, -18, 1, "dB"), ("attack", "Attack", 0.5, 50, 5, 0.5, "ms"), ("release", "Release", 10, 800, 150, 5, "ms")])
def _expander(x, thr, ratio, range, attack, release):
    return dsp.expander(x, SR, thr, ratio, range, attack, release)


# ============================================================================ time and space
REVERB_KINDS = {"Hall": "Concert Hall", "Room": "Dialogue Room", "Plate": "Plate", "Chamber": "Chamber", "Cathedral": "Cathedral", "Stadium": "Stadium"}


@effect("reverb", "Reverb", "Time and Space", [("type", "Type", list(REVERB_KINDS), "Room"), ("mix", "Mix", 0, 1, 0.2, 0.01, ""),
        ("predelay", "Pre-delay", 0, 150, 12, 1, "ms"), ("decay", "Decay scale", 0.3, 2.5, 1.0, 0.05, "x"), ("low", "Low cut", 20, 600, 180, 10, "Hz"),
        ("high", "High cut", 2000, 16000, 8000, 250, "Hz")])
def _reverb(x, type, mix, predelay, decay, low, high):
    kind = REVERB_KINDS[type]
    rt = dsp.REVERB_TYPES[kind][0] * decay
    m = x.mean(axis=1)
    wet = np.stack([dsp.reverb_wet(m, kind, 11, rt60=rt, predelay=predelay, lowcut=low, highcut=high),
                    dsp.reverb_wet(m, kind, 12, rt60=rt, predelay=predelay, lowcut=low, highcut=high)], axis=1)
    return _mix(x, wet, mix)


IR_LIBRARY = {  # synthetic impulse responses, named after the spaces they imitate (free, no licensing needed)
    "Scoring Stage-style": ("room", 1.9, 0.9, 30, 0.4), "Cathedral Nave-style": ("room", 5.5, 1.0, 45, 0.3),
    "Empty Warehouse-style": ("room", 2.6, 0.9, 25, 0.55), "Tiled Bathroom-style": ("room", 1.0, 0.2, 3, 0.1),
    "Car Interior-style": ("room", 0.14, 0.02, 0, 0.85), "Vintage Plate-style": ("room", 1.8, 0.3, 0, 0.25),
    "Spring Reverb-style": ("spring", 1.5, 0.2, 0, 0.5), "Scoring Stage Control Room-style": ("room", 0.35, 0.3, 4, 0.5),
    "Forest/Outdoor Echo-style": ("echo", 2.5, 0.8, 0, 0.7), "Small Vocal Booth-style": ("room", 0.18, 0.05, 1, 0.75),
}


def make_named_ir(name, seed=21):
    kind, rt, size, pre, damp = IR_LIBRARY[name]
    ir = dsp.make_ir(rt, size, pre, damp, seed=seed)
    if kind == "spring":                                   # metallic comb-like chirps
        comb = np.zeros(int(SR * 0.35), np.float32)
        for k in range(10):
            comb[int(k * 0.03 * SR)] = 0.8 ** k
        ir = signal.fftconvolve(ir, comb)[: len(ir)]
        ir = signal.sosfilt(dsp.butter_sos("highpass", 300, 2), ir)
    if kind == "echo":                                     # a few discrete distant reflections
        out = np.zeros(len(ir), np.float32)
        for k, t in enumerate((0.21, 0.47, 0.83, 1.31, 1.9)):
            i = int(t * SR)
            if i < len(out):
                out[i] = 0.55 ** (k + 1)
        ir = out + ir * 0.3
        ir = signal.sosfilt(dsp.butter_sos("lowpass", 3500, 2), ir)
    return (ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9)).astype(np.float32)


_ir_cache = {}


@effect("convolution", "Convolution Reverb", "Time and Space", [("ir", "Space", list(IR_LIBRARY), "Scoring Stage-style"),
        ("mix", "Mix", 0, 1, 0.25, 0.01, ""), ("predelay", "Pre-delay", 0, 150, 0, 1, "ms")])
def _convolution(x, ir, mix, predelay):
    if ir not in _ir_cache:
        _ir_cache[ir] = (make_named_ir(ir, 21), make_named_ir(ir, 22))
    m = x.mean(axis=1)
    pd = np.zeros(int(predelay * SR / 1000), np.float32)
    wet = np.stack([signal.oaconvolve(m, np.r_[pd, h])[: len(m)] for h in _ir_cache[ir]], axis=1)
    return _mix(x, wet, mix)


def _feedback_delay(sig, D, fb, process=None):
    """wet[n] = sig[n-D] + fb * wet[n-D]  (computed one delay-length chunk at a time)."""
    n = len(sig)
    wet = np.zeros_like(sig)
    prev = np.zeros((D,) + sig.shape[1:], np.float32)
    for s in range(0, n, D):
        e = min(n, s + D)
        src = np.zeros_like(prev) if s == 0 else sig[s - D:s]
        cur = src + fb * prev
        if process is not None:
            cur = process(cur)
        wet[s:e] = cur[: e - s]
        prev = cur
    return wet


@effect("delay", "Delay", "Time and Space", [("mode", "Mode", ["Mono", "Stereo", "Ping-Pong", "Tape Echo", "Slapback"], "Stereo"),
        ("time", "Time", 10, 2000, 350, 5, "ms"), ("fb", "Feedback", 0, 0.95, 0.35, 0.01, ""), ("tone", "Tone", 1000, 16000, 6000, 250, "Hz"),
        ("mix", "Mix", 0, 1, 0.25, 0.01, "")])
def _delay(x, mode, time, fb, tone, mix):
    if mode == "Slapback":
        time, fb = min(time, 120), 0.0
    D = max(16, int(time * SR / 1000))
    lp = dsp.butter_sos("lowpass", tone, 2)
    if mode == "Mono":
        wet = _feedback_delay(x.mean(axis=1, keepdims=True), D, fb)
        wet = np.repeat(wet, 2, axis=1)
    elif mode == "Ping-Pong":
        m = x.mean(axis=1)
        L = np.zeros(len(m), np.float32)
        R = np.zeros(len(m), np.float32)
        prev = np.zeros(D, np.float32)
        for k, s in enumerate(range(0, len(m), D)):
            e = min(len(m), s + D)
            src = np.zeros(D, np.float32) if s == 0 else m[s - D:s]
            cur = src + fb * prev
            (L if k % 2 == 0 else R)[s:e] = cur[: e - s]
            prev = cur
        wet = np.stack([L, R], axis=1)
    elif mode == "Tape Echo":
        def tape(c):
            return np.tanh(signal.sosfilt(lp, c, axis=0) * 1.4).astype(np.float32)
        wet = _feedback_delay(x, D, fb, tape)
    else:
        wet = _feedback_delay(x, D, fb) if mode == "Stereo" else _feedback_delay(x, D, 0.0)
        wet = signal.sosfilt(lp, wet, axis=0).astype(np.float32)
    return _mix(x, wet.astype(np.float32), mix)


def _mod_delay(x, base_ms, depth_ms, rate, phase=0.0):
    n = len(x)
    t = np.arange(n) / SR
    d = (base_ms + depth_ms * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t + phase))) * SR / 1000
    pos = np.arange(n) - d
    i0 = np.floor(pos).astype(np.int64)
    fr = (pos - i0).astype(np.float32)
    a = x[np.clip(i0, 0, n - 1)]
    b = x[np.clip(i0 + 1, 0, n - 1)]
    out = a * (1 - fr)[:, None] + b * fr[:, None]
    out[pos < 0] = 0
    return out


@effect("chorus", "Chorus", "Time and Space", [("rate", "Rate", 0.1, 5, 0.8, 0.05, "Hz"), ("depth", "Depth", 0.5, 12, 6, 0.5, "ms"), ("mix", "Mix", 0, 1, 0.4, 0.01, "")])
def _chorus(x, rate, depth, mix):
    wet = 0.5 * (_mod_delay(x, 18, depth, rate) + _mod_delay(x, 24, depth, rate * 1.13, 1.7))
    return _mix(x, wet, mix)


@effect("flanger", "Flanger", "Time and Space", [("rate", "Rate", 0.05, 5, 0.3, 0.05, "Hz"), ("depth", "Depth", 0.3, 5, 2.5, 0.1, "ms"),
        ("fb", "Feedback", 0, 0.9, 0.5, 0.01, ""), ("mix", "Mix", 0, 1, 0.5, 0.01, "")])
def _flanger(x, rate, depth, fb, mix):
    n, blk = len(x), 32
    y = np.zeros_like(x)
    base, dep = 1.0, depth
    t = np.arange(n) / SR
    d = (base + dep * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))) * SR / 1000
    for s in range(0, n, blk):
        e = min(n, s + blk)
        pos = np.arange(s, e) - d[s:e]
        i0 = np.floor(pos).astype(np.int64)
        fr = (pos - i0).astype(np.float32)[:, None]
        ya = y[np.clip(i0, 0, n - 1)] if s else np.zeros((e - s, x.shape[1]), np.float32)
        yb = y[np.clip(i0 + 1, 0, n - 1)] if s else np.zeros((e - s, x.shape[1]), np.float32)
        delayed = ya * (1 - fr) + yb * fr
        delayed[pos < 0] = 0
        y[s:e] = x[s:e] + fb * delayed
    wet = y / (1 + fb)
    return _mix(x, wet, mix)


@effect("phaser", "Phaser", "Time and Space", [("rate", "Rate", 0.05, 5, 0.4, 0.05, "Hz"), ("depth", "Depth", 0, 1, 0.8, 0.05, ""),
        ("stages", "Stages", 2, 8, 4, 2, ""), ("fb", "Feedback", 0, 0.9, 0.3, 0.05, ""), ("mix", "Mix", 0, 1, 0.5, 0.01, "")])
def _phaser(x, rate, depth, stages, fb, mix):
    n, blk = len(x), 256
    out = np.zeros_like(x)
    zi = [np.zeros((1, x.shape[1])) for _ in range(int(stages))]
    last = np.zeros((1, x.shape[1]), np.float32)
    for s in range(0, n, blk):
        e = min(n, s + blk)
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * rate * (s / SR))
        f = 300 + 3200 * (lfo ** 2) * depth
        a = (np.tan(np.pi * f / SR) - 1) / (np.tan(np.pi * f / SR) + 1)
        seg = x[s:e] + fb * last
        for k in range(int(stages)):
            seg, zi[k] = signal.lfilter([a, 1.0], [1.0, a], seg, axis=0, zi=zi[k] if zi[k].shape == (1, x.shape[1]) else None)
        last = seg[-1:] if len(seg) else last
        out[s:e] = seg
    return _mix(x, 0.5 * (x + out), mix)


# ============================================================================ EQ and filtering
def _eq_bands(x, bands):
    return dsp.parametric_eq(x, bands)


@effect("eq1", "Parametric EQ (1-band)", "EQ and Filtering", [("f", "Frequency", 20, 18000, 1000, 10, "Hz"), ("g", "Gain", -18, 18, 0, 0.5, "dB"), ("q", "Q", 0.2, 8, 1, 0.1, "")])
def _eq1(x, f, g, q):
    return _eq_bands(x, [("peak", f, g, q)])


@effect("eq3", "Parametric EQ (3-band)", "EQ and Filtering", [("lf", "Low freq", 40, 400, 120, 5, "Hz"), ("lg", "Low gain", -18, 18, 0, 0.5, "dB"),
        ("mf", "Mid freq", 300, 6000, 1500, 10, "Hz"), ("mg", "Mid gain", -18, 18, 0, 0.5, "dB"), ("mq", "Mid Q", 0.2, 8, 1, 0.1, ""),
        ("hf", "High freq", 3000, 16000, 8000, 100, "Hz"), ("hg", "High gain", -18, 18, 0, 0.5, "dB")])
def _eq3(x, lf, lg, mf, mg, mq, hf, hg):
    return _eq_bands(x, [("lowshelf", lf, lg, 0.7), ("peak", mf, mg, mq), ("highshelf", hf, hg, 0.7)])


_EQ7 = [(60, "lowshelf"), (150, "peak"), (400, "peak"), (1000, "peak"), (2500, "peak"), (6000, "peak"), (12000, "highshelf")]


@effect("eq7", "Parametric EQ (7-band)", "EQ and Filtering",
        [(f"g{i}", f"Band {i + 1} ({int(f)} Hz)", -18, 18, 0, 0.5, "dB") for i, (f, _) in enumerate(_EQ7)] + [("q", "Q (all bell bands)", 0.3, 4, 1, 0.1, "")])
def _eq7(x, q, **g):
    return _eq_bands(x, [(k, f, g[f"g{i}"], 0.7 if k != "peak" else q) for i, (f, k) in enumerate(_EQ7)])


_GEQ = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


@effect("geq", "Graphic EQ (10-band)", "EQ and Filtering", [(f"g{i}", f"{f if f < 1000 else str(f // 1000) + 'k'} Hz", -12, 12, 0, 0.5, "dB") for i, f in enumerate(_GEQ)])
def _geq(x, **g):
    return _eq_bands(x, [("peak", f, g[f"g{i}"], 1.4) for i, f in enumerate(_GEQ)])


@effect("hpf", "High-Pass Filter (low cut)", "EQ and Filtering", [("f", "Frequency", 20, 1000, 80, 5, "Hz"), ("order", "Slope (poles)", 1, 8, 4, 1, "")])
def _hpf(x, f, order):
    return dsp.highpass(x, f, SR, int(order))


@effect("lpf", "Low-Pass Filter (high cut)", "EQ and Filtering", [("f", "Frequency", 500, 20000, 12000, 50, "Hz"), ("order", "Slope (poles)", 1, 8, 4, 1, "")])
def _lpf(x, f, order):
    return dsp.lowpass(x, f, SR, int(order))


@effect("bpf", "Band-Pass Filter", "EQ and Filtering", [("f", "Centre", 100, 10000, 1000, 10, "Hz"), ("bw", "Width", 0.2, 3, 1, 0.1, "oct")])
def _bpf(x, f, bw):
    lo, hi = f / 2 ** (bw / 2), min(f * 2 ** (bw / 2), SR * 0.45)
    return dsp.apply_sos(x, signal.butter(2, [lo, hi], "bandpass", fs=SR, output="sos"))


@effect("notch", "Notch Filter", "EQ and Filtering", [("f", "Frequency", 40, 12000, 50, 1, "Hz"), ("q", "Q", 1, 80, 30, 1, "")])
def _notch(x, f, q):
    return dsp.apply_sos(x, dsp.biquad("notch", f, SR, q))


# ============================================================================ tuning and pitch
@effect("pitch", "Pitch Shifter", "Tuning and Pitch", [("semi", "Semitones", -12, 12, 0, 0.5, "st"), ("mix", "Mix", 0, 1, 1, 0.01, "")])
def _pitch(x, semi, mix):
    return _mix(x, dsp.pitch_shift(x, semi), mix)


@effect("harmonizer", "Harmonizer", "Tuning and Pitch", [("v1", "Voice 1", -12, 12, 4, 1, "st"), ("v2", "Voice 2", -12, 12, 7, 1, "st"), ("mix", "Mix", 0, 1, 0.4, 0.01, "")])
def _harmonizer(x, v1, v2, mix):
    wet = 0.5 * (dsp.pitch_shift(x, v1) + dsp.pitch_shift(x, v2))
    return _mix(x, wet, mix)


@effect("formant", "Formant Modifier", "Tuning and Pitch", [("shift", "Formant shift", 0.7, 1.4, 1.0, 0.01, "x")])
def _formant(x, shift):
    if abs(shift - 1) < 0.01:
        return x
    out = []
    for c in range(x.shape[1]):
        _, _, Z = signal.stft(x[:, c], SR, nperseg=2048, noverlap=1536)
        mag = np.abs(Z)
        env = np.maximum(signal.convolve2d(mag, np.ones((9, 1)) / 9, mode="same", boundary="symm"), 1e-9)
        bins = np.arange(Z.shape[0])
        src = np.clip(bins / shift, 0, Z.shape[0] - 1)
        env2 = np.stack([np.interp(src, bins, env[:, t]) for t in range(env.shape[1])], axis=1)
        _, y = signal.istft(Z * (env2 / env), SR, nperseg=2048, noverlap=1536)
        out.append(np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)])
    return np.stack(out, axis=1)


# ============================================================================ harmonic and distortion
@effect("saturation", "Saturation", "Harmonic and Distortion", [("type", "Type", ["Tape", "Tube", "Console"], "Tape"), ("drive", "Drive", 0.5, 12, 2.5, 0.1, "x"),
        ("mix", "Mix", 0, 1, 0.5, 0.01, "")])
def _saturation(x, type, drive, mix):
    if type == "Tape":
        wet = np.tanh(x * drive) / np.tanh(drive)
        wet = dsp.apply_sos(wet, dsp.butter_sos("lowpass", 14000, 1))
    elif type == "Tube":
        bias = 0.12
        wet = (np.tanh(x * drive + bias) - np.tanh(bias)) / np.tanh(drive)
    else:
        yc = np.clip(x * drive, -1, 1)
        wet = (yc - yc ** 3 / 3) * 1.5
    return _mix(x, np.asarray(wet, np.float32), mix)


@effect("overdrive", "Overdrive", "Harmonic and Distortion", [("drive", "Drive", 1, 30, 6, 0.5, "x"), ("tone", "Tone", 800, 12000, 4500, 100, "Hz"), ("level", "Level", -24, 6, -6, 0.5, "dB")])
def _overdrive(x, drive, tone, level):
    y = np.tanh(x * drive)
    return dsp.lowpass(y, tone, SR, 2) * float(dsp.db2lin(level))


@effect("fuzz", "Fuzz", "Harmonic and Distortion", [("gain", "Gain", 2, 60, 25, 1, "x"), ("level", "Level", -30, 0, -12, 0.5, "dB")])
def _fuzz(x, gain, level):
    return np.clip(x * gain, -0.5, 0.5) * 2 * float(dsp.db2lin(level))


@effect("bitcrush", "Bitcrusher", "Harmonic and Distortion", [("bits", "Bit depth", 2, 16, 8, 1, "bit"), ("down", "Sample-rate divider", 1, 32, 4, 1, "x"), ("mix", "Mix", 0, 1, 1, 0.01, "")])
def _bitcrush(x, bits, down, mix):
    q = 2 ** (int(bits) - 1)
    y = np.round(x * q) / q
    d = int(down)
    if d > 1:
        y = np.repeat(y[::d], d, axis=0)[: len(x)]
        y = np.pad(y, ((0, max(0, len(x) - len(y))), (0, 0)))
    return _mix(x, y, mix)


@effect("exciter", "Exciter / Enhancer", "Harmonic and Distortion", [("freq", "From", 2000, 10000, 4000, 100, "Hz"), ("amount", "Amount", 0, 1, 0.35, 0.01, "")])
def _exciter(x, freq, amount):
    hp = dsp.highpass(x, freq, SR, 2)
    harm = np.tanh(hp * 4) - hp
    return (x + amount * harm).astype(np.float32)


@effect("subharm", "Sub-harmonic Generator", "Harmonic and Distortion", [("freq", "Source below", 60, 240, 120, 5, "Hz"), ("amount", "Amount", 0, 1, 0.4, 0.01, "")])
def _subharm(x, freq, amount):
    bass = dsp.lowpass(x, freq, SR, 4)
    flip = np.cumsum(np.diff(np.r_[0, (bass[:, 0] >= 0).astype(np.int8)]) == 1) % 2     # octave-down square from zero crossings
    sub = (flip * 2 - 1).astype(np.float32)[:, None] * np.abs(bass)
    sub = dsp.lowpass(sub, freq, SR, 4)
    return (x + amount * 1.5 * sub).astype(np.float32)


# ============================================================================ utility and spatial
@effect("imager", "Stereo Imager", "Utility and Spatial", [("width", "Width", 0, 2.5, 1.0, 0.01, "x"), ("mono_below", "Mono below", 0, 400, 0, 10, "Hz")])
def _imager(x, width, mono_below):
    m, s = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    s2 = s * width
    if mono_below > 0:
        s2 = s2 - dsp.lowpass(s2[:, None], mono_below, SR, 4)[:, 0]
    return np.stack([m + s2, m - s2], axis=1)


@effect("phase", "Phase Inverter", "Utility and Spatial", [("which", "Invert", ["Both", "Left", "Right"], "Both")])
def _phase(x, which):
    y = x.copy()
    if which in ("Both", "Left"):
        y[:, 0] *= -1
    if which in ("Both", "Right"):
        y[:, 1] *= -1
    return y


@effect("gain", "Gain / Trim", "Utility and Spatial", [("db", "Gain", -48, 24, 0, 0.5, "dB")])
def _gain(x, db):
    return x * float(dsp.db2lin(db))


# ============================================================================ chains and presets
def process_chain(x, slots):
    """slots: iterable of objects/dicts with kind, params, bypass, data (JSFX source)."""
    if len(x) == 0:
        return x
    for s in slots:
        get = (lambda k, d=None: getattr(s, k, d)) if not isinstance(s, dict) else (lambda k, d=None: s.get(k, d))
        if get("bypass", False):
            continue
        kind = get("kind")
        if kind == "jsfx":
            from . import jsfx
            x = jsfx.process(x, get("data", {}).get("script", ""), get("params", {}))
        elif kind in EFFECTS:
            x = run(kind, x, get("params", {}))
    return x


def _p(kind, **kw):
    return {"kind": kind, "params": kw, "bypass": False, "data": {}}


# Plugin presets (our own .mh3fx format: JSON).  Pro Tools .tfx files are Avid-proprietary and cannot be written here.
PLUGIN_PRESETS = {
    "Vocal De-esser": _p("deess", freq=6500, thr=-36, ratio=4),
    "Room Reverb": _p("reverb", type="Room", mix=0.18),
    "Large Concert Hall Reverb": _p("reverb", type="Hall", mix=0.3, decay=1.3),
    "1/4 Note Ping-Pong Delay": _p("delay", mode="Ping-Pong", time=500, fb=0.4, mix=0.25),
    "Slapback Vocal Echo": _p("delay", mode="Slapback", time=90, mix=0.3),
    "80Hz Low-End EQ Cut": _p("hpf", f=80, order=4),
    "Presence Boost EQ": _p("eq3", lf=120, lg=0, mf=3200, mg=3.5, mq=0.9, hf=9000, hg=1.5),
    "Brickwall Limiter Peak Protector": _p("brick", ceiling=-1.0),
    "Radio/Telephone Voice Filter": _p("bpf", f=1800, bw=1.3),
    "Tape Saturation Warmth": _p("saturation", type="Tape", drive=2.2, mix=0.45),
}

TRACK_PRESETS = {  # ordered insert chains
    "Lead Vocal Chain": [_p("hpf", f=90), _p("deess", thr=-34), _p("comp", thr=-22, ratio=3.5), _p("eq3", mg=2.5, mf=3000, hg=2, hf=9500), _p("reverb", type="Plate", mix=0.12)],
    "Background Vocal Harmony": [_p("hpf", f=120), _p("comp", thr=-20, ratio=3), _p("harmonizer", mix=0.35), _p("reverb", type="Hall", mix=0.25)],
    "Acoustic Guitar Enhancer": [_p("hpf", f=80), _p("eq3", lg=-2, mf=250, mg=-2, hg=3, hf=8000), _p("comp", thr=-20, ratio=2.5), _p("exciter", amount=0.25)],
    "Heavy Rock Guitar": [_p("hpf", f=90), _p("overdrive", drive=14, tone=4200), _p("eq3", lg=2, mf=800, mg=-3, hg=2), _p("limiter", ceiling=-1)],
    "Clean Electric Bass": [_p("hpf", f=35), _p("comp", thr=-24, ratio=4, attack=20), _p("eq3", lg=2.5, lf=90, mg=-1.5, mf=400, hg=1.5, hf=5000), _p("saturation", type="Tube", mix=0.25)],
    "Kick Drum Punch": [_p("hpf", f=35), _p("gate", thr=-40), _p("eq3", lf=60, lg=4, mf=350, mg=-4, hg=3, hf=5000), _p("comp", thr=-18, ratio=4, attack=25, release=80)],
    "Snare Drum Crack": [_p("gate", thr=-38), _p("eq3", lf=200, lg=2, mf=900, mg=-2, hg=4, hf=7000), _p("comp", thr=-20, ratio=5, attack=3, release=60), _p("reverb", type="Plate", mix=0.15)],
    "Drum Bus Parallel Compressor": [_p("comp", thr=-30, ratio=8, attack=2, release=70, makeup=8, mix=0.4), _p("saturation", type="Console", mix=0.2)],
    "Master Bus Finalizer": [_p("eq3", lg=1, hg=1.5), _p("comp", thr=-18, ratio=2, attack=30, release=150), _p("imager", width=1.1), _p("maximizer", gain=3, ceiling=-0.8)],
    "Podcast Host Voice": [_p("hpf", f=85), _p("gate", thr=-52, range=-18), _p("deess", thr=-36), _p("comp", thr=-22, ratio=3.2), _p("eq3", lg=2, lf=150, mg=2, mf=3200, hg=1.5), _p("maximizer", gain=2, ceiling=-1)],
}

SESSION_TEMPLATES = {  # (track name, preset chain name or None)
    "Podcast Recording": [("Host", "Podcast Host Voice"), ("Guest", "Podcast Host Voice"), ("Music bed", None), ("SFX", None)],
    "Audiobook / Voiceover": [("Narration", "Podcast Host Voice"), ("Room tone", None)],
    "Stereo Music Mixing": [("Kick", "Kick Drum Punch"), ("Snare", "Snare Drum Crack"), ("Bass", "Clean Electric Bass"), ("Guitar", "Acoustic Guitar Enhancer"), ("Lead Vocal", "Lead Vocal Chain"), ("Backing Vocal", "Background Vocal Harmony")],
    "Dialogue Editing / ADR": [("Dialogue", "Podcast Host Voice"), ("ADR", "Podcast Host Voice"), ("Room tone", None), ("Foley", None)],
    "Foley Recording": [("Footsteps", None), ("Cloth", None), ("Props", None), ("Room tone", None)],
    "Film Scoring": [("Strings", None), ("Brass", None), ("Woodwinds", None), ("Percussion", None), ("Synth", None)],
    "Video Game Sound Design": [("Impacts", None), ("Ambience", None), ("UI", None), ("Voice", "Podcast Host Voice")],
    "Band Tracking / Multi-track": [("Kick", "Kick Drum Punch"), ("Snare", "Snare Drum Crack"), ("Bass", "Clean Electric Bass"), ("Guitar L", "Heavy Rock Guitar"), ("Guitar R", "Heavy Rock Guitar"), ("Vocal", "Lead Vocal Chain")],
    "Mastering": [("Stereo master", "Master Bus Finalizer")],
}
