"""Convolution reverb: the sound of a real space, from its impulse response.

An impulse response (IR) is the recording of one short click (or a sweep turned
into a click) in a real space: a cave, a church, a mine. It holds every echo of
that space. Convolving a sound with it plays the sound "in" that space. The FDN
in reverb.py builds a room from rules. Here the room is the real one.

A 6 s IR at 48 kHz has 288000 samples, too many for a direct convolution in real
time. The Convolver uses the usual trick, a uniform partitioned convolution
(overlap-save):

- The IR is cut into parts of PART samples, and each part is turned into a
  spectrum once, when the IR is loaded.
- Each block of PART input samples is turned into a spectrum too, and goes into
  a ring that keeps the last spectra (as many as there are parts).
- The output spectrum is the sum of each past input spectrum times its part of
  the IR (_mac below, compiled with numba). One inverse FFT gives the block.

The first part plays in the same block as its input, so with blocks of PART
samples (or a multiple) the reverb adds no delay. With other block sizes it adds
one part of delay (10.7 ms at 48 kHz), nothing you hear in a reverb.

The Convolver is linear and does not change in time, like the FDN: a held note
comes out as the same note plus the echoes of the space.
"""
import math

import numba
import numpy as np

PART = 512  # samples per IR part. Bigger = less CPU, but more delay with odd blocks


@numba.njit("void(complex64[:, ::1], complex64[:, :, ::1], int64, complex64[:, ::1])",
            cache=True, nogil=True)
def _mac(ring, h, head, acc):
    """acc[c] = sum over p of ring[head - p] * h[c, p] (the ring wraps around)."""
    n_parts, n_bins = ring.shape
    n_ch = h.shape[0]
    for c in range(n_ch):
        for k in range(n_bins):
            acc[c, k] = 0.0
    for p in range(n_parts):
        r = head - p
        if r < 0:
            r += n_parts
        for c in range(n_ch):
            for k in range(n_bins):
                acc[c, k] += ring[r, k] * h[c, p, k]


class Convolver:
    """Plays a mono input through a (frames, channels) impulse response.

    process(x) takes a mono block of any size and returns (frames, channels).
    """

    def __init__(self, ir: np.ndarray, part: int = PART):
        ir = np.asarray(ir, dtype=np.float64)
        if ir.ndim == 1:
            ir = ir[:, None]
        n, self.channels = ir.shape
        self.part = part
        self.n_parts = max(1, math.ceil(n / part))
        padded = np.zeros((self.n_parts * part, self.channels))
        padded[:n] = ir
        # (channels, parts, part) -> each part zero-padded to 2*part -> spectra
        parts = padded.T.reshape(self.channels, self.n_parts, part)
        self.h = np.ascontiguousarray(
            np.fft.rfft(parts, n=2 * part, axis=-1).astype(np.complex64))
        self.ring = np.zeros((self.n_parts, part + 1), dtype=np.complex64)
        self.acc = np.zeros((self.channels, part + 1), dtype=np.complex64)
        self.head = 0
        self.last = np.zeros(2 * part)            # the last two input parts
        self.fifo_in = np.zeros(0)                # input waiting for a full part
        self.fifo_out = np.zeros((0, self.channels))
        self.delayed = False                      # True once blocks are not whole parts

    def _run_part(self, seg: np.ndarray) -> np.ndarray:
        """One part of input (part,) -> one part of output (part, channels)."""
        p = self.part
        self.last[:p] = self.last[p:]
        self.last[p:] = seg
        self.head = (self.head + 1) % self.n_parts
        self.ring[self.head] = np.fft.rfft(self.last).astype(np.complex64)
        _mac(self.ring, self.h, self.head, self.acc)
        # Overlap-save: the second half of the inverse FFT is the clean output.
        return np.fft.irfft(self.acc, n=2 * p, axis=-1)[:, p:].T

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        frames, p = x.shape[0], self.part
        if not self.delayed and frames % p == 0:
            # Blocks of whole parts: no fifo, no delay.
            if frames == p:
                return self._run_part(x)
            return np.concatenate([self._run_part(x[i:i + p]) for i in range(0, frames, p)])
        if not self.delayed:
            # From now on the output is one part late. With that delay there is
            # always enough output ready, whatever the block sizes.
            self.delayed = True
            self.fifo_out = np.zeros((p, self.channels))
        buf = np.concatenate([self.fifo_in, x])
        n_full = buf.shape[0] // p
        outs = [self.fifo_out] + [self._run_part(buf[i * p:(i + 1) * p]) for i in range(n_full)]
        self.fifo_in = buf[n_full * p:]
        out = np.concatenate(outs)
        self.fifo_out = out[frames:]
        return out[:frames]


if __name__ == "__main__":
    # Offline self-test: the output must be the exact convolution (to float32
    # precision), with any block size, and fast enough for real time.
    import time

    from scipy.signal import fftconvolve

    SR = 48000
    rng = np.random.default_rng(0)
    n_ir = int(2.0 * SR) + 123
    decay = np.exp(-np.arange(n_ir) / (0.4 * SR))[:, None]
    ir = rng.standard_normal((n_ir, 2)) * decay * 0.01
    x = rng.standard_normal(int(3.0 * SR))
    ref = np.stack([fftconvolve(x, ir[:, c])[:x.size] for c in range(2)], axis=1)
    ok = True
    for block in (512, 1024, 256, 300, 777):
        cv = Convolver(ir)
        # All blocks the same size, like an audio callback (pad the end).
        xp = np.concatenate([x, np.zeros(-x.size % block)])
        y = np.concatenate([cv.process(xp[i:i + block]) for i in range(0, xp.size, block)])
        y = y[:x.size]
        # Find the delay (0 or one part) and compare.
        lag = 0 if np.abs(y[:PART] - ref[:PART]).max() < 1e-4 * np.abs(ref).max() else PART
        diff = y[lag:] - ref[:x.size - lag]
        err = 20 * np.log10(np.sqrt(np.mean(diff ** 2)) / np.sqrt(np.mean(ref ** 2)))
        good = err < -100.0 and (lag == 0) == (block % PART == 0)
        ok &= good
        print(f"block {block:4d}: delay {lag:3d} samples, error {err:6.1f} dB"
              f" {'ok' if good else 'BAD'}")

    for secs in (3.0, 6.0, 10.0):
        cv = Convolver(rng.standard_normal((int(secs * SR), 2)) * 0.01)
        blk = rng.standard_normal(512)
        cv.process(blk)
        t0 = time.perf_counter()
        n = 200
        for _ in range(n):
            cv.process(blk)
        ms = (time.perf_counter() - t0) / n * 1e3
        print(f"IR {secs:4.1f} s stereo: {ms:.2f} ms per 512 block (budget 10.7 ms)")
    print("all good" if ok else "FAILED")
