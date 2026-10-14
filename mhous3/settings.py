"""Settings = a plain dict (easy to save as JSON).  PRESETS only list what differs from DEFAULTS."""
import json
from pathlib import Path

from .dsp import REVERB_TYPES

SAMPLE_RATES = [44100, 48000, 88200, 96000, 176400, 192000]
BIT_DEPTHS = {"16-bit": 16, "24-bit": 24, "32-bit float": 32}
OUT_CHANNELS = ["Mono", "Stereo", "Dolby Surround (Lt/Rt stereo)", "Dolby Digital 5.1"]

# label -> (extension, max channels, ffmpeg audio args or None for soundfile-native)
FORMATS = {
    "WAV": (".wav", 8),
    "FLAC": (".flac", 8),
    "MP3 (320 kbps)": (".mp3", 2),
    "AAC / M4A": (".m4a", 6),
    "Dolby Digital AC-3": (".ac3", 6),
    "Dolby Digital Plus E-AC-3": (".eac3", 6),
    "Video + new audio (MP4)": (".mp4", 6),
    "Video + new audio (MKV)": (".mkv", 6),
}
CHANNEL_COUNT = {"Mono": 1, "Stereo": 2, "Dolby Surround (Lt/Rt stereo)": 2, "Dolby Digital 5.1": 6}
LOUDNESS_TARGETS = {
    "Cinema / streaming dialogue (-27 LUFS)": -27.0,
    "Broadcast TV (-24 LUFS)": -24.0,
    "Broadcast EBU R128 (-23 LUFS)": -23.0,
    "Podcast / YouTube (-16 LUFS)": -16.0,
    "Loud online (-14 LUFS)": -14.0,
    "Custom (use slider)": None,
}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".mts", ".m2ts", ".mxf", ".webm", ".mpg", ".mpeg", ".wmv", ".m4v"}


def formats_for(channels_label):
    n = CHANNEL_COUNT[channels_label]
    return [k for k, (_, mx) in FORMATS.items() if n <= mx]


def _defaults():
    d = dict(
        # input / repair
        input_mode="Auto", declip="Auto", declip_max_run=64, hum="Off", lowcut=80.0, highcut=0.0,
        # noise + room (basic macros)
        nr_amount=60, nr_engine="Auto", nr_sens=2.0, derev_amount=35, derev_rt60=0.6,
        clarity=35, warmth=25, polish=40,
        # noise gate / expander
        gate_on=True, gate_thr=-52.0, gate_ratio=2.0, gate_range=-10.0, gate_attack=6.0, gate_release=180.0,
        # de-esser
        deess_on=True, deess_freq=6500.0, deess_thr=-34.0, deess_ratio=3.0,
        # compressor
        comp_on=True, comp_thr=-22.0, comp_ratio=2.5, comp_attack=14.0, comp_release=120.0,
        comp_knee=8.0, comp_makeup=0.0,
        # saturation / width
        sat_on=False, sat_drive=1.8, sat_mix=25, width=100,
        # reverb
        reverb_type="Dialogue Room", reverb_mix=0, reverb_predelay=-1.0, reverb_damp=-1.0, reverb_rt60=0.0,
        reverb_lowcut=180.0, reverb_highcut=8000.0,
        # surround
        center_level=80, surround_level=60, lfe_level=0,
        # fast-track macros
        volume_boost=0.0, room_expansion=0,
        # output
        out_sr=48000, out_bits=24, out_channels="Stereo", out_format="WAV", loudness_on=True,
        loudness_target="Podcast / YouTube (-16 LUFS)", lufs_custom=-16.0, ceiling=-1.0,
    )
    kinds = ["lowshelf", "peak", "peak", "peak", "peak", "highshelf"]
    freqs = [110, 250, 800, 2800, 5200, 11000]
    qs = [0.7, 0.9, 0.9, 1.0, 1.0, 0.7]
    for i in range(6):
        d[f"eq{i + 1}_kind"] = kinds[i]
        d[f"eq{i + 1}_f"] = float(freqs[i])
        d[f"eq{i + 1}_g"] = 0.0
        d[f"eq{i + 1}_q"] = qs[i]
    return d


DEFAULTS = _defaults()

PRESETS = {
    "Cinema Dialogue (Hollywood)": dict(nr_amount=70, derev_amount=55, derev_rt60=0.7, clarity=55, warmth=40,
                                        polish=70, comp_ratio=3.0, reverb_type="Dialogue Room", reverb_mix=12,
                                        sat_on=True, sat_mix=22, lowcut=85.0,
                                        loudness_target="Broadcast TV (-24 LUFS)", ceiling=-2.0),
    "Camera Mic Rescue": dict(nr_amount=80, derev_amount=60, derev_rt60=0.7, clarity=45, warmth=30, polish=45,
                              gate_on=True, hum="Off"),
    "Clipped Recorder Rescue": dict(declip="On", declip_max_run=96, nr_amount=55, derev_amount=30, clarity=40,
                                    warmth=35, polish=45, deess_on=True),
    "Podcast / Voice-over": dict(nr_amount=65, derev_amount=40, clarity=50, warmth=45, polish=55,
                                 comp_ratio=3.2, loudness_target="Podcast / YouTube (-16 LUFS)"),
    "Interview (noisy location)": dict(nr_amount=90, derev_amount=45, clarity=50, warmth=20, polish=40,
                                       gate_range=-14.0, lowcut=110.0),
    "Big Cinema Hall (add space)": dict(nr_amount=60, derev_amount=45, clarity=45, warmth=40, polish=70,
                                        reverb_type="Cinema Hall", reverb_mix=30, sat_on=True,
                                        out_channels="Dolby Digital 5.1", out_format="Dolby Digital AC-3",
                                        loudness_target="Broadcast TV (-24 LUFS)", ceiling=-2.0),
    "Phone / Zoom Call Fix": dict(nr_amount=75, derev_amount=50, clarity=60, warmth=15, polish=30, lowcut=120.0,
                                  highcut=0.0),
    "Gentle (keep it natural)": dict(nr_amount=30, derev_amount=15, clarity=15, warmth=10, polish=15,
                                     gate_on=False, comp_ratio=1.8, deess_on=False),
    "Loudness Only": dict(nr_amount=0, derev_amount=0, clarity=0, warmth=0, polish=0, gate_on=False,
                          deess_on=False, comp_on=False, lowcut=0.0, declip="Off"),
}


def preset_settings(name):
    s = dict(DEFAULTS)
    s.update(PRESETS.get(name, {}))
    return s


def target_lufs(s):
    v = LOUDNESS_TARGETS.get(s["loudness_target"])
    return float(s["lufs_custom"]) if v is None else v


def reverb_choices():
    return list(REVERB_TYPES)


# ---------- persistence (user presets + last settings)
def user_dir():
    base = Path.home() / "AppData" / "Roaming" if (Path.home() / "AppData").exists() else Path.home() / ".config"
    d = base / "MHous3 Audio Studio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_user_presets():
    try:
        return json.loads((user_dir() / "presets.json").read_text("utf-8"))
    except Exception:
        return {}


def save_user_presets(p):
    (user_dir() / "presets.json").write_text(json.dumps(p, indent=2), "utf-8")


PRESET_DESC = {
    "Cinema Dialogue (Hollywood)": "Polished film-dialogue sound: clean, warm and present.",
    "Camera Mic Rescue": "Fixes hiss and room echo from built-in camera mics.",
    "Clipped Recorder Rescue": "Repairs distorted, clipped recordings.",
    "Podcast / Voice-over": "Rich, even voice for podcasts and narration.",
    "Interview (noisy location)": "Strong noise removal for streets, cafes and events.",
    "Big Cinema Hall (add space)": "Grand 5.1 cinema space with reverb.",
    "Phone / Zoom Call Fix": "Clears up recorded calls and meetings.",
    "Gentle (keep it natural)": "Light touch-up that keeps it natural.",
    "Loudness Only": "No cleaning - just match the loudness.",
}
