import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mhous3 import dsp  # noqa: E402

SR = 48000
rng = np.random.default_rng(7)


def speechlike(sec=12):
    """Syllable-modulated harmonic bursts with pauses - crude but good enough for DSP checks."""
    n = SR * sec
    t = np.arange(n) / SR
    env = np.clip(np.sin(2 * np.pi * 2.2 * t), 0, 1) ** 2 * (np.sin(2 * np.pi * 0.35 * t) > -0.4)
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.8 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 20))
    return (0.25 * x / np.abs(x).max() * env).astype(np.float32), env


def snr_db(clean, test, env):
    mask = env > 0.5
    sig = np.mean(test[mask] ** 2)
    noise = np.mean(test[env < 0.01] ** 2)
    return 10 * np.log10(sig / (noise + 1e-12))


def test_loudness_matches_ffmpeg():
    x, _ = speechlike(15)
    x = np.stack([x, x * 0.8], axis=1)
    mine = dsp.integrated_lufs(x)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.wav"
        sf.write(p, x, SR, subtype="FLOAT")
        ff = sys.argv[1] if len(sys.argv) > 1 else "ffmpeg"
        out = subprocess.run([ff, "-hide_banner", "-i", str(p), "-af", "ebur128", "-f", "null", "-"],
                             capture_output=True, text=True).stderr
    ref = float(out.split("Integrated loudness:")[-1].split("I:")[1].split("LUFS")[0])
    print(f"LUFS mine={mine:.2f} ffmpeg={ref:.2f}")
    assert abs(mine - ref) < 0.3


def test_normalize_hits_target_and_ceiling():
    x, _ = speechlike()
    y, _ = dsp.normalize_loudness(np.stack([x, x], 1), -16, -1.0)
    l = dsp.integrated_lufs(y)
    tp = dsp.true_peak_db(y)
    print(f"normalised: {l:.2f} LUFS, true peak {tp:.2f} dB")
    assert abs(l + 16) < 0.6 and tp <= -0.9


def test_true_peak_guard():
    t = np.arange(SR * 3)
    x = np.stack([np.sin(2 * np.pi * t / 4 + np.pi / 4)] * 2, 1).astype(np.float32)        # fs/4 sine: samples at -3 dB, true peak 0 dB
    assert dsp.true_peak_db(x) > -0.1 and 20 * np.log10(np.abs(x).max()) < -2.9
    y, _ = dsp.normalize_loudness(x * 0.5, -10, -1.0)
    tp = dsp.true_peak_db(y); print(f"true peak after guard: {tp:.2f} dBTP"); assert tp <= -0.95, tp
    z = dsp.limiter(x, SR, -3.0, 6.0, true_peak=True); assert dsp.true_peak_db(z) <= -2.9

def test_denoise_improves_snr():
    clean, env = speechlike()
    noisy = clean + 0.02 * rng.standard_normal(len(clean)).astype(np.float32)
    out = dsp.spectral_denoise(noisy[:, None], amount_db=30)[:, 0]
    a, b = snr_db(clean, noisy, env), snr_db(clean, out, env)
    print(f"denoise SNR {a:.1f} -> {b:.1f} dB")
    assert b > a + 8


def test_dereverb_cuts_tail():
    clean, env = speechlike()
    ir = dsp.make_ir(0.9, 0.5, 5, 0.4, seed=3)
    wet = signal.oaconvolve(clean, ir)[: len(clean)]
    out = dsp.dereverb(wet[:, None], strength=0.7, rt60=0.9)[:, 0]
    gap = (env < 0.01)
    gap[: SR] = False
    before, after = np.mean(wet[gap] ** 2), np.mean(out[gap] ** 2)
    print(f"gap energy (reverb tail) reduced by {10 * np.log10(before / after):.1f} dB")
    assert after < before * 0.5


def test_declip_restores_peaks():
    clean, _ = speechlike()
    ref = clean * 4.6
    loud = np.clip(ref, -1, 1)
    fixed, runs, frac = dsp.declip(loud[:, None])
    err_before = np.mean((ref - loud) ** 2)
    err_after = np.mean((ref - fixed[:, 0]) ** 2)
    print(f"declip: {runs} runs fixed ({frac:.1%} clipped), error {err_before:.4f} -> {err_after:.4f}")
    assert runs > 10 and err_after < err_before * 0.6


def test_limiter_never_exceeds_ceiling():
    x = np.stack([rng.standard_normal(SR * 5) * 0.6] * 2, 1).astype(np.float32)
    y = dsp.limiter(x, SR, -2.0)
    assert np.abs(y).max() <= dsp.db2lin(-2.0) + 1e-4


def test_dynamics_speed_and_effect():
    x, _ = speechlike(60)
    x = x[:, None]
    t = time.time()
    y = dsp.compressor(x, SR, -24, 4, 10, 100)
    z = dsp.expander(y, SR, -50, 2, -12, 5, 150)
    w = dsp.deesser(z, SR, 6500, -35, 3)
    dt = time.time() - t
    print(f"compressor+expander+deesser on 60 s audio: {dt:.1f}s")
    assert dt < 20
    assert (np.percentile(np.abs(y), 99) / (np.percentile(np.abs(y), 50) + 1e-9)) < (
        np.percentile(np.abs(x), 99) / (np.percentile(np.abs(x), 50) + 1e-9)) + 1e-6


def test_reverb_and_surround():
    x, _ = speechlike(4)
    for kind in dsp.REVERB_TYPES:
        w = dsp.reverb_wet(x, kind, seed=1)
        assert w.shape == x.shape and np.isfinite(w).all()
    wet = np.stack([dsp.reverb_wet(x, "Cinema Hall", 1), dsp.reverb_wet(x, "Cinema Hall", 2)], 1)
    x51 = dsp.upmix_51(np.stack([x, x], 1), wet * 0.2, wet * 0.2)
    assert x51.shape[1] == 6
    ltrt = dsp.lt_rt_encode(x51)
    assert ltrt.shape[1] == 2 and np.isfinite(ltrt).all()
    assert dsp.downmix_to_stereo(x51).shape[1] == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
