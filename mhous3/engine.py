"""Decode -> render (the cleanup chain) -> export.  No GUI code in here, so it is easy to test."""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import dsp
from .dsp import SR
from .paths import NOWIN, find_tool
from .settings import FORMATS, VIDEO_EXT, target_lufs

SURROUND_W = [1, 1, 1, 0, 1.41, 1.41]


class Cancelled(Exception):
    pass


class EngineError(RuntimeError):
    pass


def tool_status():
    return {"ffmpeg": bool(find_tool("ffmpeg")), "deepfilter": bool(find_tool("deep-filter"))}


def _ffmpeg():
    p = find_tool("ffmpeg")
    if not p:
        raise EngineError("ffmpeg was not found. Reinstall MHous3 Audio Studio (it ships with ffmpeg).")
    return p


def _run(cmd, cancel=None):
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, creationflags=NOWIN)
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    if r.returncode != 0:
        raise EngineError(f"{Path(str(cmd[0])).name} failed: {r.stderr.decode(errors='replace')[-600:]}")
    return r


# ----------------------------------------------------------------------------- decode
def decode(path, cancel=None):
    """Any audio/video file -> float32 (n, 2) at 48 kHz."""
    r = _run([_ffmpeg(), "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", SR, "-f", "f32le", "pipe:1"], cancel)
    a = np.frombuffer(r.stdout, dtype=np.float32)
    if a.size < 2:
        raise EngineError("No audio stream found in that file.")
    return a[: a.size // 2 * 2].reshape(-1, 2).copy()


# ----------------------------------------------------------------------------- render
def _is_dual_mono(x):
    if x.shape[1] < 2:
        return True
    side = np.mean((x[:, 0] - x[:, 1]) ** 2)
    mid = np.mean((x[:, 0] + x[:, 1]) ** 2) + 1e-12
    return side / mid < 1e-5


def _fit(y, n):
    if len(y) >= n:
        return y[:n]
    return np.pad(y, ((0, n - len(y)),) + ((0, 0),) * (y.ndim - 1))


def _deepfilter(x, exe, amount, cancel):
    out = np.zeros_like(x)
    pf = ["--pf"] if amount >= 75 else []
    atten = [] if amount >= 100 else ["-a", f"{6 + amount / 100 * 94:.0f}"]
    for c in range(x.shape[1]):
        v = x[:, c]
        peak = float(np.max(np.abs(v))) + 1e-9
        g = 0.9 / peak
        tmp = Path(tempfile.mkdtemp(prefix="mh3_df_"))
        try:
            inp, od = tmp / "in.wav", tmp / "out"
            od.mkdir()
            sf.write(inp, v * g, SR, subtype="PCM_16")
            _run([exe] + pf + atten + ["-o", od, inp], cancel)
            y, _ = sf.read(od / "in.wav", dtype="float32")
            out[:, c] = _fit(y if y.ndim == 1 else y[:, 0], len(v)) / g
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return out


def _denoise(x, s, info, cancel):
    eng, amt = s["nr_engine"], float(s["nr_amount"])
    exe = find_tool("deep-filter")
    if eng != "Built-in" and exe:
        try:
            y = _deepfilter(x, exe, amt, cancel)
            info["nr_engine"] = "DeepFilterNet (AI)"
            return y
        except Cancelled:
            raise
        except Exception as e:  # fall back rather than fail the whole job
            info["warning"] = f"AI denoiser failed, used built-in instead ({str(e)[:120]})"
    info["nr_engine"] = "Built-in spectral" + (" (AI tool not found)" if eng == "AI (DeepFilterNet)" and not exe else "")
    return dsp.spectral_denoise(x, amount_db=6 + amt / 100 * 34, sensitivity=float(s["nr_sens"]))


def _eq_bands(s):
    bands = [(s[f"eq{i}_kind"], s[f"eq{i}_f"], s[f"eq{i}_g"], s[f"eq{i}_q"]) for i in range(1, 7)]
    c, w, p = s["clarity"] / 100, s["warmth"] / 100, s["polish"] / 100
    bands += [("peak", 300, -2.5 * c, 1.0), ("peak", 3300, 3.5 * c, 0.9), ("highshelf", 7500, 1.5 * c, 0.7),
              ("lowshelf", 170, 3.0 * w, 0.7), ("highshelf", 9500, -1.5 * w, 0.7), ("highshelf", 11000, 2.5 * p, 0.6)]
    return bands


def _wet(m, s, seed):
    pd = None if s["reverb_predelay"] < 0 else s["reverb_predelay"]
    dm = None if s["reverb_damp"] < 0 else s["reverb_damp"]
    return dsp.reverb_wet(m, s["reverb_type"], seed, rt60=s["reverb_rt60"] or None, predelay=pd, damping=dm,
                          lowcut=s["reverb_lowcut"], highcut=s["reverb_highcut"])


def _width(x, w):
    if x.shape[1] != 2 or abs(w - 100) < 1:
        return x
    m, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2 * (w / 100)
    return np.stack([m + side, m - side], axis=1).astype(np.float32)


def _ambience(m, re):
    """Short 'room expansion' ambience (stereo pair) for the FastTrack knob."""
    g = 0.55 * re ** 1.2
    return g * np.stack([dsp.reverb_wet(m, "Small Room", 6, lowcut=250, highcut=9000),
                         dsp.reverb_wet(m, "Small Room", 7, lowcut=250, highcut=9000)], axis=1)


def _layout(x, s):
    mode = s["out_channels"]
    mix = float(s["reverb_mix"]) / 100
    re = float(s.get("room_expansion", 0)) / 100
    gw = mix ** 1.4
    m = dsp.downmix_mono(x)[:, 0]
    if mode == "Mono":
        y = dsp.downmix_mono(x)
        if mix > 0:
            y = y + gw * _wet(m, s, 0)[:, None]
        if re > 0:
            y = y + _ambience(m, re).mean(axis=1, keepdims=True)
        return y
    st = np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x
    st = _width(st, float(s["width"]) + 40 * re)
    amb = _ambience(m, re) if re > 0 else None
    if mode == "Stereo":
        if mix > 0:
            st = st + gw * np.stack([_wet(m, s, 0), _wet(m, s, 1)], axis=1)
        if amb is not None:
            st = st + amb
        return st.astype(np.float32)
    zeros = np.zeros((len(st), 2), np.float32)
    wf = np.stack([_wet(m, s, 0), _wet(m, s, 1)], axis=1) * gw * 0.7 if mix > 0 else zeros
    ws = np.stack([_wet(m, s, 2), _wet(m, s, 3)], axis=1) * gw if mix > 0 else zeros
    if amb is not None:
        wf, ws = wf + amb, ws + amb * 1.2
    x51 = dsp.upmix_51(st, wf, ws, center=s["center_level"] / 100, surround=s["surround_level"] / 100,
                       lfe=s["lfe_level"] / 100)
    return dsp.lt_rt_encode(x51) if mode.startswith("Dolby Surround") else x51


REPAIR_KEYS = ("input_mode", "declip", "declip_max_run", "hum", "lowcut", "highcut", "nr_amount", "nr_engine",
               "nr_sens", "derev_amount", "derev_rt60")


def _repair(audio, s, info, step, cancel):
    x = np.asarray(audio, dtype=np.float32)
    if s["input_mode"] == "Mono" or (s["input_mode"] == "Auto" and _is_dual_mono(x)):
        x = dsp.downmix_mono(x)
    info["lufs_in"] = dsp.integrated_lufs(x)
    info["peak_in_db"] = float(dsp.lin2db(np.max(np.abs(x)))) if x.size else -120.0

    step(0.04, "Checking for clipping")
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    hot = peak > 0.95 and np.count_nonzero(np.abs(x) >= peak * 0.995) >= 8
    if s["declip"] == "On" or (s["declip"] == "Auto" and hot):
        step(0.08, "Repairing clipped peaks")
        x, runs, frac = dsp.declip(x, max_run=int(s["declip_max_run"]))
        info["declip_runs"], info["clipped_fraction"] = runs, frac

    step(0.14, "Removing hum and rumble")
    if s["hum"] != "Off":
        x = dsp.remove_hum(x, float(s["hum"]))
    x = dsp.highpass(x, float(s["lowcut"]))
    x = dsp.lowpass(x, float(s["highcut"]))

    if s["nr_amount"] > 0:
        step(0.22, "Removing noise")
        x = _denoise(x, s, info, cancel)
    if s["derev_amount"] > 0:
        step(0.42, "Removing room echo")
        x = dsp.dereverb(x, s["derev_amount"] / 100, float(s["derev_rt60"]))
    return x


def render(audio, s, progress=None, cancel=None, cache=None, token=None):
    """audio (n,2) float32 @48k -> (processed (n,ch_out) float32, info dict).

    `cache` (a dict) + `token` (changes whenever the source audio changes) let repeated renders skip the
    slow repair stages when only tone / space / loudness settings changed."""
    def step(f, msg):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if progress:
            progress(f, msg)

    key = (token, len(audio), tuple(s[k] for k in REPAIR_KEYS)) if (cache is not None and token is not None) else None
    if key is not None and cache.get("key") == key:
        x, info = cache["x"], dict(cache["info"])
        step(0.5, "Using cached repair")
    else:
        info = {}
        x = _repair(audio, s, info, step, cancel)
        if key is not None:
            cache.clear()
            cache.update(key=key, x=x, info=dict(info))

    step(0.52, "Shaping tone")
    x = dsp.parametric_eq(x, _eq_bands(s))
    if s["gate_on"]:
        x = dsp.expander(x, SR, s["gate_thr"], s["gate_ratio"], s["gate_range"], s["gate_attack"], s["gate_release"])
    if s["deess_on"]:
        x = dsp.deesser(x, SR, s["deess_freq"], s["deess_thr"], s["deess_ratio"])

    step(0.62, "Dynamics and polish")
    if s["comp_on"]:
        x = dsp.compressor(x, SR, s["comp_thr"], s["comp_ratio"], s["comp_attack"], s["comp_release"],
                           s["comp_knee"], s["comp_makeup"])
    p = s["polish"] / 100
    if p > 0:
        par = dsp.compressor(x, SR, -34, 8, 2, 70, 6, 10)
        x = (x + 0.35 * p * par).astype(np.float32)
    if s["sat_on"]:
        x = dsp.saturate(x, s["sat_drive"], s["sat_mix"] / 100)

    step(0.72, "Building channel layout and space")
    y = _layout(x, s)

    step(0.86, "Matching loudness")
    w = SURROUND_W if y.shape[1] == 6 else None
    boost = float(s.get("volume_boost", 0.0))
    if s["loudness_on"]:
        y, _ = dsp.normalize_loudness(y, target_lufs(s) + boost, float(s["ceiling"]), w)
    else:
        y = dsp.limiter(y * float(dsp.db2lin(boost)), SR, float(s["ceiling"]))
    info["lufs_out"] = dsp.integrated_lufs(y, w)
    info["peak_out_db"] = float(dsp.lin2db(np.max(np.abs(y)))) if y.size else -120.0
    info["channels"] = y.shape[1]
    step(1.0, "Done")
    return y.astype(np.float32), info


# ----------------------------------------------------------------------------- export
def default_output_path(src, s, out_dir=None, suffix="_MHous3"):
    ext = FORMATS[s["out_format"]][0]
    src = Path(src)
    if ext in (".mp4", ".mkv") and src.suffix.lower() not in VIDEO_EXT:
        ext = ".wav"
    d = Path(out_dir) if out_dir else src.parent
    d.mkdir(parents=True, exist_ok=True)
    dst, n = d / f"{src.stem}{suffix}{ext}", 2
    while dst.exists():
        dst, n = d / f"{src.stem}{suffix}{n}{ext}", n + 1
    return dst


def _prepare_out(y, s):
    """Resample to the chosen output rate and dither when going down to 16 bit."""
    out_sr, bits = int(s.get("out_sr", SR)), int(s.get("out_bits", 24))
    if out_sr != SR:
        from math import gcd
        from scipy import signal as _sg
        g = gcd(out_sr, SR)
        y = _sg.resample_poly(y, out_sr // g, SR // g, axis=0).astype(np.float32)
    if bits == 16:
        y = dsp.tpdf_dither(y, 16)
    return y, out_sr, bits


def export(y, dst, s, src_path=None, cancel=None):
    dst = Path(dst)
    ext = dst.suffix.lower()
    ch = y.shape[1]
    y, out_sr, bits = _prepare_out(y, s)
    wav_sub = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bits, "PCM_24")
    if ext == ".wav":
        sf.write(dst, y, out_sr, subtype=wav_sub)
        return
    if ext == ".flac":
        sf.write(dst, y, out_sr, format="FLAC", subtype="PCM_16" if bits == 16 else "PCM_24")
        return
    ff = _ffmpeg()
    tmp = Path(tempfile.mkdtemp(prefix="mh3_exp_"))
    try:
        wav = tmp / "mix.wav"
        sf.write(wav, y, out_sr, subtype="PCM_24")
        surround_flag = ["-dsur_mode", "on"] if s["out_channels"].startswith("Dolby Surround") else []
        br = {1: ("96k", "128k", "128k", "160k"), 2: ("192k", "256k", "320k", "256k"),
              6: ("448k", "384k", "640k", "640k")}[ch]  # ac3 / aac / eac3 / mp3(n/a)
        if ext == ".mp3":
            codec = ["-c:a", "libmp3lame", "-b:a", "320k", "-ar", str(min(out_sr, 48000))]
        elif ext == ".m4a":
            codec = ["-c:a", "aac", "-b:a", br[1]]
        elif ext == ".ac3":
            codec = ["-c:a", "ac3", "-b:a", br[0]] + surround_flag
        elif ext == ".eac3":
            codec = ["-c:a", "eac3", "-b:a", br[2]]
        elif ext in (".mp4", ".mkv"):
            codec = (["-c:a", "ac3", "-b:a", br[0]] + surround_flag) if ch > 2 or surround_flag else \
                    ["-c:a", "aac", "-b:a", br[1]]
        else:
            raise EngineError(f"Unsupported export format {ext}")
        if ext in (".mp4", ".mkv"):
            if not src_path or Path(src_path).suffix.lower() not in VIDEO_EXT:
                raise EngineError("'Video + new audio' needs a video source file.")
            cmd = [ff, "-y", "-v", "error", "-i", src_path, "-i", wav, "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "copy"] + codec + ["-shortest", dst]
        else:
            cmd = [ff, "-y", "-v", "error", "-i", wav] + codec + [dst]
        _run(cmd, cancel)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def process_file(src, dst, s, progress=None, cancel=None):
    if progress:
        progress(0.0, "Reading file")
    audio = decode(src, cancel)
    y, info = render(audio, s, (lambda f, m: progress(0.05 + 0.85 * f, m)) if progress else None, cancel)
    if progress:
        progress(0.92, "Saving")
    export(y, dst, s, src, cancel)
    if progress:
        progress(1.0, "Done")
    return info
