"""Auto Clean: measure a recording and recommend settings (noise, echo, clipping, hum, tone, dynamics)."""
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from . import dsp
from .settings import DEFAULTS

SR = 48000


def _band(fw, P, lo, hi):
    return float(P[(fw >= lo) & (fw < hi)].sum()) + 1e-20


def _tone_score(db, bin_hz, base):
    """How strongly a mains tone stands out: (fundamental dB above neighbours, best harmonic dB above neighbours).
    Voices have almost nothing at exactly 50/60 Hz, so the fundamental must be present to count as hum."""
    out = []
    for k in (1, 2, 3):
        i = int(round(base * k / bin_hz))
        if i - 13 < 0 or i + 13 >= len(db):
            out.append(0.0)
            continue
        nb = np.r_[db[i - 12:i - 3], db[i + 4:i + 13]]
        out.append(float(db[i - 1:i + 2].max() - np.median(nb)))
    return out[0], max(out[1:])


def analyze(audio):
    """audio (n,2|1) float @48k -> feature dict.  Fast: most work is done at 16 kHz."""
    x = np.asarray(audio, np.float32)
    x = x.mean(axis=1) if x.ndim == 2 else x
    n = len(x)
    if n < SR // 2 or float(np.max(np.abs(x))) < 1e-4:
        return {"valid": False}
    f = {"valid": True, "duration": n / SR}
    peak = float(np.max(np.abs(x)))
    near = int(np.count_nonzero(np.abs(x) >= peak * 0.995))
    f.update(peak_db=float(dsp.lin2db(peak)), clip_count=near, clip_frac=near / n, clipped=bool(peak > 0.95 and near >= 8))
    f["lufs"] = dsp.integrated_lufs(x)

    y = signal.resample_poly(x, 1, 3).astype(np.float32)
    fs = 16000
    hop = 320
    m = len(y) // hop
    fr = y[:m * hop].reshape(m, hop).astype(np.float64)
    lvl = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    noise, speech = float(np.percentile(lvl, 10)), float(np.percentile(lvl, 95))
    f.update(noise_floor_db=noise, speech_db=speech, snr_db=speech - noise)
    active = lvl[lvl > noise + 6]
    f["dyn_db"] = float(np.percentile(active, 95) - np.percentile(active, 25)) if len(active) > 20 else 10.0

    yw = y[: fs * 600]
    fw, P = signal.welch(yw, fs, nperseg=8192)
    db = 10 * np.log10(P + 1e-14)
    bin_hz = fs / 8192
    (a1, a2), (b1, b2) = _tone_score(db, bin_hz, 50), _tone_score(db, bin_hz, 60)
    is50, is60 = (a1 > 10 and a2 > 5), (b1 > 10 and b2 > 5)
    f["hum_hz"] = 50 if (is50 and (not is60 or a1 >= b1)) else 60 if is60 else 0
    tot = P.sum() + 1e-20
    f["rumble_db"] = float(10 * np.log10(_band(fw, P, 0, 70) / tot))
    f["bright_db"] = float(10 * np.log10(_band(fw, P, 4000, 7500) / _band(fw, P, 300, 3000)))
    f["sibil_db"] = float(10 * np.log10(_band(fw, P, 5000, 7500) / _band(fw, P, 1000, 4000)))
    f["low_db"] = float(10 * np.log10(_band(fw, P, 80, 250) / _band(fw, P, 300, 3000)))

    # reverb time from the decay after loud syllable ends
    eh = 160
    k = len(y) // eh
    env = 10 * np.log10(np.mean(y[:k * eh].reshape(k, eh).astype(np.float64) ** 2, axis=1) + 1e-12)
    env = uniform_filter1d(env, 3)
    cand = np.where((env >= speech - 8) & (env == maximum_filter1d(env, 11)))[0]
    slopes = []
    for i in cand:
        if i + 22 >= k:
            continue
        seg = env[i + 2:i + 22]
        ok = seg > noise + 4
        cnt = len(seg) if ok.all() else int(np.argmin(ok))
        if cnt >= 8:
            sl = np.polyfit(np.arange(cnt) * 0.01, seg[:cnt], 1)[0]
            if sl < 0:
                slopes.append(sl)
    f["rt60"] = float(np.clip(60 / abs(np.percentile(slopes, 25)), 0.15, 2.5)) if len(slopes) >= 3 else 0.3
    return f


def _level(v, cuts, names):
    for c, nme in zip(cuts, names):
        if v < c:
            return nme
    return names[-1]


def recommend(f, keep_output=None):
    """feature dict -> (settings dict, human-readable notes, scenario name)"""
    s = dict(DEFAULTS)
    s.update(declip="Auto", reverb_mix=0, room_expansion=0, volume_boost=0.0, sat_on=False)
    notes = []
    if not f.get("valid"):
        notes.append("The audio is silent or very short, so safe default settings were chosen.")
        return _keep(s, keep_output), notes, "Default"

    snr = f["snr_db"]
    s["nr_amount"] = 90 if snr < 8 else 80 if snr < 14 else 65 if snr < 22 else 45 if snr < 32 else 25
    notes.append(f"Background noise: {_level(snr, (10, 18, 28), ('very strong', 'strong', 'moderate', 'low'))} "
                 f"(about {snr:.0f} dB between speech and noise) - noise reduction {s['nr_amount']} %.")
    rt = f["rt60"]
    s["derev_amount"] = 10 if rt < 0.25 else 25 if rt < 0.4 else 45 if rt < 0.7 else 60 if rt < 1.0 else 72
    s["derev_rt60"] = float(np.clip(round(rt, 2), 0.3, 1.5))
    notes.append(f"Room echo: {_level(rt, (0.3, 0.55, 0.9), ('very little', 'some', 'noticeable', 'heavy'))} "
                 f"(about {rt:.1f} s) - echo removal {s['derev_amount']} %.")
    if f["clipped"]:
        s["declip"] = "On"
        s["clarity"] = 28
        notes.append(f"Clipping found ({f['clip_count']} flattened samples) - the peaks will be rebuilt.")
    else:
        notes.append("No clipping found.")
    if f["hum_hz"]:
        s["hum"] = str(f["hum_hz"])
        notes.append(f"Mains hum detected at {f['hum_hz']} Hz - it will be removed.")
    if f["rumble_db"] > -18:
        s["lowcut"] = 110.0
        notes.append("Low-frequency rumble found - low cut raised to 110 Hz.")
    if f["bright_db"] < -38 and not f["clipped"]:
        s["clarity"], s["warmth"] = 58, 15
        notes.append("The recording sounds dull or muffled - clarity increased.")
    elif not f["clipped"]:
        s["clarity"] = 38
    if f["low_db"] < -14:
        s["warmth"] = 38
        notes.append("The voice sounds thin - warmth added.")
    if f["sibil_db"] > -17:
        s["deess_thr"], s["deess_ratio"] = -40.0, 4.0
        notes.append("Harsh 's' sounds - stronger de-essing.")
    dr = f["dyn_db"]
    s["comp_ratio"] = 3.5 if dr > 30 else 2.8 if dr > 20 else 2.0
    s["polish"] = 45
    if snr < 15:
        s["gate_on"], s["gate_range"] = True, -14.0
    s["loudness_target"] = "Podcast / YouTube (-16 LUFS)"
    notes.append(f"Loudness will be matched to -16 LUFS (it measures {f['lufs']:.0f} LUFS now).")
    if f["clipped"]:
        name = "Clipped recording"
    elif snr < 12:
        name = "Noisy recording"
    elif rt > 0.7:
        name = "Echoey room"
    elif f["bright_db"] < -38:
        name = "Muffled / phone-quality recording"
    else:
        name = "Fairly clean speech"
    return _keep(s, keep_output), notes, name


def _keep(s, keep):
    if keep:
        for k in ("out_channels", "out_format", "loudness_on", "loudness_target", "lufs_custom", "ceiling"):
            s[k] = keep[k]
    return s
