import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mhous3 import fx, dsp
SR = 48000
rng = np.random.default_rng(5)

def sig(sec=3):
    t = np.arange(SR * sec) / SR
    x = 0.4 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0) + 0.05 * rng.standard_normal(len(t))
    return np.stack([x, x * 0.9], 1).astype(np.float32)

def test_every_effect_runs():
    x = sig(); n = 0
    for eid, e in fx.EFFECTS.items():
        t = time.time(); y = fx.run(eid, x); dt = time.time() - t
        assert y.shape == x.shape and np.isfinite(y).all(), eid
        n += 1; print(f"  {e['cat']:26s} {e['name']:32s} {dt:5.2f}s")
    print(n, "effects OK"); assert n >= 33

def test_behaviour():
    x = np.zeros((SR * 2, 2), np.float32); x[1000:1200] = 0.8
    d = fx.run("delay", x, dict(mode="Mono", time=250, fb=0.0, mix=1.0, tone=16000))
    lag = np.argmax(np.abs(d[2000:, 0])) + 2000 - 1000
    assert abs(lag - 0.25 * SR) < 400, lag                                    # echo arrives 250 ms later
    pp = fx.run("delay", x, dict(mode="Ping-Pong", time=250, fb=0.6, mix=1.0))
    assert np.abs(pp[:, 0]).max() > 0.1 and np.abs(pp[:, 1]).max() > 0.1      # repeats alternate sides
    s = sig(); tone = np.stack([0.5 * np.sin(2 * np.pi * 220 * np.arange(SR * 2) / SR)] * 2, 1).astype(np.float32)
    h = fx.run("hpf", tone, dict(f=2000, order=4)); assert np.abs(h[SR:]).max() < 0.02
    b = fx.run("bitcrush", s, dict(bits=3, down=1, mix=1.0)); assert len(np.unique(np.round(b[:, 0], 4))) <= 9
    l = fx.run("limiter", s * 4, dict(ceiling=-6, look=6)); assert np.abs(l).max() <= dsp.db2lin(-6) + 1e-3
    c = fx.run("comp", s, dict(thr=-30, ratio=8, mix=1.0)); assert c.std() < s.std()
    p = fx.run("pitch", np.stack([np.sin(2 * np.pi * 440 * np.arange(SR * 1.5) / SR)] * 2, 1).astype(np.float32), dict(semi=12, mix=1.0))
    f0 = np.argmax(np.abs(np.fft.rfft(p[:, 0] * np.hanning(len(p))))) * SR / len(p); assert abs(f0 - 880) < 25, f0
    im = fx.run("imager", s, dict(width=0.0)); assert np.allclose(im[:, 0], im[:, 1], atol=1e-5)
    ph = fx.run("phase", s, dict(which="Both")); assert np.allclose(ph, -s)
    v = fx.run("reverb", x, dict(type="Hall", mix=0.5)); assert np.abs(v[SR:]).max() > 1e-4                   # tail present
    cv = fx.run("convolution", x, dict(ir="Tiled Bathroom-style", mix=0.5)); assert np.isfinite(cv).all()

def test_chain_and_presets():
    s = sig(2)
    y = fx.process_chain(s, [fx.PLUGIN_PRESETS["80Hz Low-End EQ Cut"], fx.PLUGIN_PRESETS["Brickwall Limiter Peak Protector"]])
    assert y.shape == s.shape
    for name, chain in fx.TRACK_PRESETS.items():
        out = fx.process_chain(s, chain); assert np.isfinite(out).all() and out.shape == s.shape, name
    for name, p in fx.PLUGIN_PRESETS.items():
        out = fx.process_chain(s, [p]); assert np.isfinite(out).all(), name
    byp = [dict(fx.PLUGIN_PRESETS["Tape Saturation Warmth"], bypass=True)]
    assert np.allclose(fx.process_chain(s, byp), s)
    assert len(fx.PLUGIN_PRESETS) == 10 and len(fx.TRACK_PRESETS) == 10 and len(fx.SESSION_TEMPLATES) == 9 and len(fx.IR_LIBRARY) == 10

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
