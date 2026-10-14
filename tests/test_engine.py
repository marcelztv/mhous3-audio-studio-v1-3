import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from mhous3 import dsp, engine, settings  # noqa: E402
from test_dsp import speechlike  # noqa: E402

SR = 48000
rng = np.random.default_rng(3)


def messy_clip(sec=10):
    clean, _ = speechlike(sec)
    ir = dsp.make_ir(0.8, 0.5, 5, 0.4, seed=5)
    wet = signal.oaconvolve(clean, ir)[: len(clean)]
    noisy = wet * 2.0 + 0.02 * rng.standard_normal(len(clean)).astype(np.float32) + 0.01 * np.sin(
        2 * np.pi * 50 * np.arange(len(clean)) / SR).astype(np.float32)
    clipped = np.clip(noisy * 3.0, -1, 1)
    return np.stack([clipped, np.clip(clipped * 0.9 + 0.01 * rng.standard_normal(len(clip := clipped)), -1, 1)], 1).astype(np.float32)


def ffinfo(path):
    ff = engine._ffmpeg()
    return subprocess.run([ff, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr


def main():
    clip = messy_clip()
    print("tools:", engine.tool_status())
    out_dir = Path(tempfile.mkdtemp(prefix="mh3test_"))
    # every preset
    for name in settings.PRESETS:
        s = settings.preset_settings(name)
        s["out_channels"], s["out_format"] = "Stereo", "WAV"
        t = time.time()
        y, info = engine.render(clip, s)
        tgt = settings.target_lufs(s)
        print(f"{name:32s} {time.time() - t:5.1f}s  in {info['lufs_in']:.1f} -> out {info['lufs_out']:.1f} LUFS "
              f"(target {tgt}), peak {info['peak_out_db']:.1f} dB, declip={info.get('declip_runs', '-')}, "
              f"nr={info.get('nr_engine', '-')}")
        assert np.isfinite(y).all() and y.shape == (len(clip), 2)
        if s["loudness_on"] and s["comp_on"] is not None and name != "Loudness Only":
            assert abs(info["lufs_out"] - tgt) < 1.0, name
    # channel modes + formats
    wav = out_dir / "in.wav"
    sf.write(wav, clip, SR)
    combos = [("Mono", "WAV"), ("Mono", "MP3 (320 kbps)"), ("Stereo", "FLAC"), ("Stereo", "AAC / M4A"),
              ("Stereo", "Dolby Digital AC-3"), ("Dolby Surround (Lt/Rt stereo)", "Dolby Digital AC-3"),
              ("Dolby Digital 5.1", "WAV"), ("Dolby Digital 5.1", "Dolby Digital AC-3"),
              ("Dolby Digital 5.1", "Dolby Digital Plus E-AC-3"), ("Dolby Digital 5.1", "AAC / M4A")]
    for ch, fmt in combos:
        s = settings.preset_settings("Cinema Dialogue (Hollywood)")
        s.update(out_channels=ch, out_format=fmt, reverb_mix=25)
        dst = engine.default_output_path(wav, s, out_dir)
        info = engine.process_file(str(wav), dst, s)
        meta = ffinfo(dst)
        line = [l for l in meta.splitlines() if "Audio:" in l][0].strip() if not str(dst).endswith(".wav") else \
            f"wav {sf.info(dst).channels}ch"
        print(f"{ch:32s} {fmt:28s} -> {dst.name}  {line[:90]}  out {info['lufs_out']:.1f} LUFS")
        assert dst.exists() and dst.stat().st_size > 1000
    # video export
    ff = engine._ffmpeg()
    vid = out_dir / "clip.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=10:r=25", "-i", str(wav),
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", str(vid)], check=True)
    for ch, fmt in [("Stereo", "Video + new audio (MP4)"), ("Dolby Digital 5.1", "Video + new audio (MKV)")]:
        s = settings.preset_settings("Podcast / Voice-over")
        s.update(out_channels=ch, out_format=fmt)
        dst = engine.default_output_path(vid, s, out_dir)
        engine.process_file(str(vid), dst, s)
        meta = ffinfo(dst)
        print(f"video {fmt}: ", [l.strip()[:80] for l in meta.splitlines() if "Stream #" in l])
        assert "Video: h264" in meta
    print("ENGINE OK")


if __name__ == "__main__":
    main()
