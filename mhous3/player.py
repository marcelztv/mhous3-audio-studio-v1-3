"""Sample-accurate A/B player: switch between original and cleaned audio instantly, with a tiny crossfade."""
import threading

import numpy as np

try:
    import sounddevice as sd
    SD_ERROR = None
except Exception as e:  # PortAudio missing or no audio backend
    sd = None
    SD_ERROR = str(e)


class ABPlayer:
    FADE = 480  # samples (10 ms)

    def __init__(self, sr=48000):
        self.sr = sr
        self.a = self.b = None
        self.src = "B"
        self._prev = "B"
        self._fade = 0
        self.pos = 0
        self.playing = False
        self.loop = None          # (start, end) in samples, relative to the buffers
        self.gain_a = 1.0
        self.peak = 0.0
        self.stream = None
        self.blocksize = 1024
        self.lock = threading.Lock()

    @property
    def available(self):
        return sd is not None

    def set_audio(self, a, b):
        with self.lock:
            self.a, self.b = a, b
            if b is None:
                self.src = "A"
            self.pos = min(self.pos, self.length())

    def length(self):
        return 0 if self.a is None else len(self.a)

    def ensure_stream(self):
        if sd is None:
            raise RuntimeError("Audio playback is unavailable: " + str(SD_ERROR))
        if self.stream is None:
            self.stream = sd.OutputStream(samplerate=self.sr, channels=2, dtype="float32",
                                          blocksize=self.blocksize, callback=self._callback)
            self.stream.start()

    def play(self):
        if self.a is None:
            return
        self.ensure_stream()
        if self.pos >= self.length():
            self.pos = 0
        self.playing = True

    def pause(self):
        self.playing = False

    def stop(self):
        self.playing = False
        self.pos = self.loop[0] if self.loop else 0

    def seek(self, pos):
        self.pos = int(max(0, min(pos, self.length())))

    def switch(self, src):
        if src != self.src and (src == "A" or self.b is not None):
            self._prev, self.src, self._fade = self.src, src, self.FADE

    def _chunk(self, which, pos, n):
        if which == "A" or self.b is None:
            return self.a[pos:pos + n] * self.gain_a
        return self.b[pos:pos + n]

    def _callback(self, out, frames, _time, _status):
        out[:] = 0
        if not self.playing or self.a is None:
            self.peak *= 0.8
            return
        with self.lock:
            pos, filled = self.pos, 0
            end = self.loop[1] if self.loop else self.length()
            while filled < frames:
                if pos >= end:
                    if self.loop:
                        pos, end = self.loop[0], self.loop[1]
                        if end <= pos:
                            self.playing = False
                            break
                    else:
                        self.playing = False
                        break
                take = min(frames - filled, end - pos)
                new = self._chunk(self.src, pos, take)
                if self._fade > 0:
                    old = self._chunk(self._prev, pos, take)
                    w = np.clip((self._fade - np.arange(take)) / self.FADE, 0, 1)[:, None]
                    new = new * (1 - w) + old * w
                    self._fade = max(0, self._fade - take)
                out[filled:filled + take] = new
                filled += take
                pos += take
            self.pos = pos
        np.clip(out, -1.0, 1.0, out=out)
        self.peak = float(np.max(np.abs(out))) if frames else 0.0

    def close(self):
        self.playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
