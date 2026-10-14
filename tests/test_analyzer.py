import sys
from pathlib import Path
import numpy as np
from scipy import signal
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from mhous3 import analyzer, dsp
from test_dsp import speechlike
from test_engine import messy_clip
rng = np.random.default_rng(11)
SR = 48000

def st(x): return np.stack([x, x], 1).astype(np.float32)

def test_all():
    clean, _ = speechlike(14)
    clean_a = st(clean * 1.5)
    noisy = st(clean * 1.5 + 0.03 * rng.standard_normal(len(clean)).astype(np.float32))
    ir = dsp.make_ir(1.0, 0.6, 5, 0.4, seed=2)
    echo = st(signal.oaconvolve(clean, ir)[: len(clean)] * 3)
    hum = st(clean * 1.5 + 0.03 * np.sin(2 * np.pi * 50 * np.arange(len(clean)) / SR).astype(np.float32)
             + 0.015 * np.sin(2 * np.pi * 100 * np.arange(len(clean)) / SR).astype(np.float32))
    clipped = messy_clip(12)
    res = {}
    for name, a in (("clean", clean_a), ("noisy", noisy), ("echo", echo), ("hum", hum), ("clipped", clipped)):
        f = analyzer.analyze(a); s, notes, scen = analyzer.recommend(f)
        res[name] = (f, s, scen)
        print(f"{name:8s} snr={f['snr_db']:5.1f} rt={f['rt60']:.2f} hum={f['hum_hz']} clip={f['clipped']} -> NR {s['nr_amount']} "
              f"echo {s['derev_amount']} declip={s['declip']} hum={s['hum']} | {scen}")
    assert res["noisy"][1]["nr_amount"] > res["clean"][1]["nr_amount"]
    assert res["echo"][0]["rt60"] > res["clean"][0]["rt60"]
    assert res["echo"][1]["derev_amount"] >= res["clean"][1]["derev_amount"]
    assert res["hum"][0]["hum_hz"] == 50 and res["hum"][1]["hum"] == "50"
    assert res["clipped"][0]["clipped"] and res["clipped"][1]["declip"] == "On"
    assert not res["clean"][0]["clipped"]
    assert analyzer.analyze(np.zeros((SR * 2, 2), np.float32))["valid"] is False
    print("ANALYZER OK")

if __name__ == "__main__":
    test_all()
