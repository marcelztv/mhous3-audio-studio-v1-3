# MHous3 Audio Studio

Free, offline audio rescue and cinema-style finishing for Windows. Everything (ffmpeg + DeepFilterNet AI denoiser) is bundled in one installer. No accounts, no cloud, no paid parts.

## Get the installer (.exe) - one-time, free, ~10 minutes, no coding

### Option A - GitHub builds it for you (easiest, nothing to install)
1. Create a free account at github.com, click **New repository** (name it e.g. `mhous3`, **Public**).
2. On the new repo page click **uploading an existing file**, drag in EVERYTHING from this folder
   (keep the folder structure, including the hidden `.github` folder), click **Commit changes**.
   Tip: easier is `Add file > Upload files` with the unzipped folder's contents, or use GitHub Desktop.
3. Open the **Actions** tab > **Build Windows installer** > **Run workflow**.
4. After ~10-15 minutes open the finished run and download **MHous3-Audio-Studio-Setup** (a zip with the installer .exe).
5. Double-click the installer. Done - it installs like any normal app.

### Option B - build on your own Windows PC
Install Python 3.11+ and Inno Setup 6 (both free), then double-click `build_windows.bat`.
The installer appears in `installer_output\`.

Windows may show "unknown publisher" (the installer isn't code-signed, which costs money). Click **More info > Run anyway**.

## What it does
* Camera-mic / clipped-recorder rescue: declipping, hum removal, AI + built-in noise reduction, de-reverb
* Cinema finishing: voice EQ, gate/expander, de-esser, compressor, parallel compression, saturation, 10 reverb rooms
* Output: Mono, Stereo, Dolby Surround (Lt/Rt), Dolby Digital 5.1 (AC-3 / E-AC-3), WAV/FLAC/MP3/AAC, or MP4/MKV video with the new audio
* LUFS loudness matching (-27 / -24 / -23 / -16 / -14 / custom) + look-ahead limiter and true-peak guard
* A/B preview with loudness matching, waveform + spectrogram, selection editing (trim/cut/silence/fade/gain/normalize/reverse), undo/redo
* Single-file Studio, Batch queue, saved user presets, drag & drop

## Developing / testing
```
pip install -r requirements.txt
python tests/test_dsp.py            # signal-processing checks (needs ffmpeg on PATH or bin/)
python main.py
```
Put `ffmpeg.exe` and `deep-filter.exe` in a `bin/` folder to run from source with everything enabled.

## Version 1.1.0 (Stage 1)
* Three workspaces: FastTrack, Precision, Full Studio (Full Studio's multi-track console, JSFX and AAF/OMF are planned for later stages)
* Auto Clean (analyses the audio, then you pick a preset or use the recommendation), Volume Boost, Room Expansion, preset cards
* Keep-changes-or-start-from-original popup when moving to a more advanced workspace after an automatic clean
* Dark / Light / System themes, animated splash screen, skippable first-run tutorial (replay from the Tutorial button)
* Customizable keyboard shortcuts per workspace and dockable panel layouts (saved automatically)
* Live scrolling spectrogram in every workspace; background rendering with result caching for a responsive interface

## Digital signing (optional, needs a paid certificate)
1. Buy a code-signing certificate issued to M Hous3 Media and export it as a .pfx file.
2. Convert it to text: `[Convert]::ToBase64String([IO.File]::ReadAllBytes("cert.pfx"))` (PowerShell).
3. In GitHub: Settings > Secrets and variables > Actions > New repository secret. Add `WINDOWS_CERT_PFX_BASE64` (the text) and `WINDOWS_CERT_PASSWORD`.
4. Run the workflow again. The program and installer are signed automatically.
The licence text is in `LICENSE.txt` (a draft: have it reviewed by a lawyer).

## Version 1.2.0 (Stage 2: Precision multi-track)
* Multi-track timeline in Precision and Full Studio: tracks, clips, drag to move, trim edges (standard and loop trim), fade handles
  (linear / equal power / equal gain, automatic crossfades on overlap), clip gain line, clip mute/lock, clip pitch shift
* Edit modes Slip / Shuffle / Grid / Spot, grid and nudge values, Cut / Copy / Paste / Clear / Duplicate / Repeat,
  Separate at playhead / selection / grid / transients, batch fades
* Per-track fader, pan, mute/solo, groups, low cut, 3-band EQ, gate and compression, volume automation lanes with breakpoints
* The timeline mix feeds the same master cleanup chain, so A/B listening, Auto Clean and export still work on the mix
* Stems export (one WAV per track, same length), project save/open (.mh3p) with automatic media consolidation into a project folder
* Waveform editor (single file) is still there as the second tab
Not yet: pitch-bend automation nodes, time trim/scrub trim, pan and send automation lanes, playlist comping, elastic audio (planned for later stages).

## Version 1.3.0 (Stage 3: mixer, effects, scripting, interchange)
* Mixer tab: track / bus / VCA / master strips, faders, pan, aux sends (pre/post), routing, insert chains, automation mode (Read / Off acts; Write / Latch / Touch need the live engine, planned)
* 33 built-in effects with auto-generated controls, 10 plugin presets (.mh3fx data), 10 track presets, 9 session templates, 10 synthetic impulse responses
* JSFX scripting subset (REAPER-style): sliders, @init / @slider / @block / @sample, mem[] buffers, loops, math functions (user functions and gmem not supported)
* Macros, session settings (time / timecode / bars|beats ruler, frame rate, tempo, buffer size), musical grid values
* Session export: REAPER .rpp, AAF (audio embedded, cuts and gaps), CMX3600 EDL (drop-frame aware), Final Cut XML; import: .rpp and Final Cut XML
* Meters: momentary / short-term / integrated LUFS, loudness range, true peak; limiter now guards true (inter-sample) peaks
* Export options: sample rate 44.1-192 kHz, 16 / 24 / 32-bit float, dither for 16-bit
Not available: AAX / AU / VST3 hosting, Melodyne, certified Dolby Atmos, OMF, Pro Tools preset formats, live automation recording, pitch-correction, elastic audio.
Run all self-tests: `python tests/test_dsp.py`, `test_fx.py`, `test_jsfx.py`, `test_project.py`, `test_mixer.py`, `test_interchange.py`, `test_analyzer.py`, `test_engine.py` and the GUI tests (`QT_QPA_PLATFORM=offscreen`).
