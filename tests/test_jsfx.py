import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mhous3 import jsfx, fx
SR = 48000
rng = np.random.default_rng(3)

def noise(sec=2, amp=0.3):
    return (amp * rng.standard_normal((SR * sec, 2))).astype(np.float32)

def test_examples_run():
    x = noise()
    for name, src in jsfx.EXAMPLES.items():
        t = time.time(); y = jsfx.process(x, src); dt = time.time() - t
        assert y.shape == x.shape and np.isfinite(y).all(), name
        print(f"  {name:24s} {dt:5.2f}s for 2 s stereo ({2 / dt:.1f}x real time)")

def test_behaviour():
    x = noise()
    g = jsfx.process(x, jsfx.EXAMPLES["Gain (dB)"], {"slider1": 6.0}); assert abs(np.abs(g).max() / np.abs(x).max() - 10 ** (6 / 20)) < 0.01
    t = jsfx.process(np.ones((SR * 2, 2), np.float32) * 0.5, jsfx.EXAMPLES["Tremolo"], {"slider1": 4.0, "slider2": 1.0}); assert t[:, 0].min() < 0.05 and t[:, 0].max() > 0.45
    lp = jsfx.process(x, jsfx.EXAMPLES["One-pole low-pass"], {"slider1": 300.0}); assert lp.std() < x.std() * 0.4
    cr = jsfx.process(np.clip(x, -1, 1), jsfx.EXAMPLES["Lo-fi crusher"], {"slider1": 3.0, "slider2": 1.0}); assert len(np.unique(np.round(cr[:, 0], 4))) <= 9
    imp = np.zeros((SR * 2, 2), np.float32); imp[1000] = 1.0
    e = jsfx.process(imp, jsfx.EXAMPLES["Echo (memory buffer)"], {"slider1": 300.0, "slider2": 0.0, "slider3": 1.0})
    lag = np.argmax(np.abs(e[3000:, 0])) + 3000 - 1000; assert abs(lag - 0.3 * SR) < 5, lag

def test_language_features():
    x = np.ones((1000, 2), np.float32) * 0.1
    src = "desc:t\\n@sample\\nacc = 0; loop(4, acc += 1;); i = 0; while (i < 3) ( acc += 10; i += 1; ); spl0 = acc/100; spl1 = (acc > 30 ? 1 : 0) * 0.5 + abs(-0.25) + min(1,2)*0 + sqrt(4)*0 + 2^3*0;".replace("\\n", "\n")
    y = jsfx.process(x, src); assert abs(y[0, 0] - 0.34) < 1e-6 and abs(y[0, 1] - 0.75) < 1e-6, (y[0])
    for bad in ("desc:t\n@sample\nspl0 = foo(1);", "desc:t\n@sample\nspl0 = ;", "desc:t\n@sample\nspl0 = 1 +;"):
        try:
            jsfx.process(x, bad); raise AssertionError("should fail: " + bad)
        except jsfx.JsfxError as e:
            pass
    inj = "desc:t\n@sample\nspl0 = __import__('os');"
    try:
        jsfx.process(x, inj); raise AssertionError("injection must fail")
    except jsfx.JsfxError:
        pass

def test_in_chain_and_speed():
    x = noise(1)
    slot = {"kind": "jsfx", "params": {"slider1": 6.0}, "bypass": False, "data": {"script": jsfx.EXAMPLES["Gain (dB)"]}}
    y = fx.process_chain(x, [slot]); assert abs(np.abs(y).max() / np.abs(x).max() - 1.995) < 0.02
    long = noise(10); t = time.time(); jsfx.process(long, jsfx.EXAMPLES["One-pole low-pass"]); dt = time.time() - t
    print(f"  10 s stereo through a JSFX low-pass: {dt:.1f}s"); assert dt < 30

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
