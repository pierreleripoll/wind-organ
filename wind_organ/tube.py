"""A blown pipe: the wind finds a tube and it starts to sing.

This is the "organic" end of organ_organic. The clean organ is a sum of sines.
Here the sound is made by the air itself, like wind in a crack of a cave or in
a ruin that happens to sound like an organ pipe.

The model is a loop, one per rank of the pipe:

    wind noise --+--> [jet: tanh] --> air column (delay, one period) --+--> out
                 ^                                                   |
                 +---- loop gain g ---- [mode filter: bandpass] <----+

- The delay is the air column. It is one period long, so the loop rings at the
  note and at all its harmonics (the modes of an open pipe).
- The bandpass in the loop picks the mode that speaks. At its centre it has
  zero phase and unity gain, so the pipe is in tune with no correction. To
  overblow, its centre slides up to the octave and the 2nd mode takes over.
- The loop gain grows with the air pressure. Under 1 the loop only rings: the
  wind noise resonates in the tube, a breathy, pitched hiss. Over 1 the loop
  catches and oscillates: the pipe sings. The tanh keeps the tone from growing
  for ever. Its small offset (the jet hits the edge a bit off centre) makes the
  even harmonics.
- The wind noise is always in the loop, so even the sung tone breathes and
  wavers, the way a real pipe in moving air does. It is lowpassed at a few
  harmonics of the note first: the breath of the pipe is a deep "hoo" around
  the note. The broad hiss is already in the wind layer.
- The sound out is the change of the air in the tube (its first difference),
  like the sound a pipe mouth radiates. This lifts the harmonics by 6 dB per
  octave: the pipe is brighter, closer to an organ flue pipe than to an ocarina.

Efficiency: the loop has one recursion per sample, which is very slow in plain
Python. So the loop is compiled with numba (_loop below): it runs sample by
sample like C, about 15 times faster than numpy calls on small chunks. It is
compiled when this module is imported (and cached on disk), never in the audio
callback.
"""
import math

import numba
import numpy as np
from scipy.signal import lfilter

TUBE_BUF = 4096           # delay buffer per rank (a power of two): pipes down to ~12 Hz
TUBE_F_MIN = 15.0
NOISE_RMS = 0.132         # RMS of the engine's pink wind noise, to bring it to 1
# Loop gain from the air pressure: g = TUBE_G0 + TUBE_G1 * pressure. The pipe
# starts to sing where g passes 1, at pressure (1 - TUBE_G0) / TUBE_G1.
TUBE_G0 = 0.8
TUBE_G1 = 0.75
TUBE_Q = 1.0              # Q of the mode filter: lower = more harmonics, less stable pitch
TUBE_NOISE = 0.15         # wind noise into the loop, per unit of pressure
TUBE_NOISE_HARM = 4.0     # the noise is lowpassed at this harmonic of the note
TUBE_BIAS = (0.3, 0.5)    # offset of the jet (random per pipe): the even harmonics.
                          # Over ~0.6 the pipe does not catch any more.


def _bandpass(fc: float, q: float, sr: int):
    """RBJ bandpass, 0 dB and zero phase at fc."""
    w0 = 2.0 * math.pi * fc / sr
    alpha = math.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha
    return (np.array([alpha, 0.0, -alpha]) / a0,
            np.array([a0, -2.0 * math.cos(w0), 1.0 - alpha]) / a0)


# The signature makes numba compile at import time, not at the first call (that
# would block the audio for a second).
@numba.njit("int64(float64[:, ::1], int64, float64[:, ::1], float64[::1], float64[::1],"
            " float64[::1], float64[::1], float64[::1], float64[::1], float64[:, ::1],"
            " float64, float64[:, ::1])", cache=True, nogil=True)
def _loop(buf, pos, zi, d0, d1, b, a, gain, press, nz, bias, out):
    """The pipe loop, sample by sample, for each rank: read the air column one
    period back, mode filter, add the wind, jet (tanh), write it back.

    buf: (ranks, TUBE_BUF) the air columns. pos: write position of the first
    sample. zi: (ranks, 2) state of the mode filter (the same form as scipy's
    lfilter). d0, d1: delay of each rank at the start and the end of the block.
    b, a: the mode filter. gain, press: (frames,) loop gain and pressure.
    nz: (ranks, frames) wind noise. out: (ranks, frames) is filled. Returns
    the next write position."""
    ranks, frames = nz.shape
    mask = buf.shape[1] - 1
    tb = math.tanh(bias)
    for r in range(ranks):
        z0, z1 = zi[r, 0], zi[r, 1]
        dd = d1[r] - d0[r]
        for n in range(frames):
            w = pos + n
            rp = w - (d0[r] + dd * (n / frames))
            i0 = int(math.floor(rp))
            fr = rp - i0
            y0 = buf[r, i0 & mask]
            x = y0 + (buf[r, (i0 + 1) & mask] - y0) * fr
            h = z0 + b[0] * x
            z0 = z1 + x * b[1] - h * a[1]
            z1 = x * b[2] - h * a[2]
            s = math.tanh(gain[n] * h + press[n] * nz[r, n] + bias) - tb
            buf[r, w & mask] = s
            out[r, n] = s
        zi[r, 0], zi[r, 1] = z0, z1
    return (pos + frames) & mask


class Tube:
    """One blown pipe with `ranks` loops (its detuned copies).

    process() takes the frequency of each rank at the start and the end of the
    block, the pressure at the start and the end, and the wind noise. It returns
    the sound of each rank, (ranks, frames), roughly in -1..1."""

    def __init__(self, sr: int, ranks: int, rng: np.random.Generator):
        self.sr = sr
        self.rng = rng
        self.bias = rng.uniform(*TUBE_BIAS)   # each pipe (each crack) is a bit different
        self.ranks = 0
        self.buf = np.zeros((0, TUBE_BUF))
        self.zi = np.zeros((0, 2))
        self.noise_zi = np.zeros((0, 1))
        self.last = np.zeros(0)  # last sample of each rank (for the radiated output)
        self.pos = 0
        self.f_prev = None
        self.set_ranks(ranks)

    def set_ranks(self, ranks: int) -> None:
        """Keep the loops that stay. New loops start silent."""
        if ranks == self.ranks:
            return
        k = min(ranks, self.ranks)
        buf = np.zeros((ranks, TUBE_BUF))
        zi = np.zeros((ranks, 2))
        nzi = np.zeros((ranks, 1))
        last = np.zeros(ranks)
        buf[:k], zi[:k], nzi[:k], last[:k] = self.buf[:k], self.zi[:k], self.noise_zi[:k], self.last[:k]
        self.buf, self.zi, self.noise_zi, self.last, self.ranks = buf, zi, nzi, last, ranks
        self.f_prev = None

    def process(self, freqs: np.ndarray, p0: float, p1: float, noise: np.ndarray,
                mode: float = 1.0) -> np.ndarray:
        """freqs: (ranks,) frequency of each rank at the end of the block (it
        glides from the last block's). p0, p1: air pressure at the start and
        the end of the block. noise: (frames, 2) wind noise. mode: the centre of
        the mode filter in harmonics of the note (1 = the note, 2 = overblown
        to the octave). Returns (ranks, frames)."""
        sr = self.sr
        frames = noise.shape[0]
        f1 = np.clip(np.asarray(freqs, dtype=np.float64), TUBE_F_MIN, sr * 0.45)
        f0 = f1 if self.f_prev is None else self.f_prev
        self.f_prev = f1
        d0, d1 = sr / f0, sr / f1                       # delay (samples) = one period

        # The mode filter is shared by the ranks (they are a fraction of a Hz
        # apart): centred on the mean note, times `mode`.
        fc = min(sr * 0.45, float(np.mean(f1)) * max(1.0, mode))
        b, a = _bandpass(fc, TUBE_Q, sr)

        # Each rank breathes its own blend of the left and right wind noise,
        # lowpassed around the first harmonics of the note.
        theta = np.arange(self.ranks) * (math.pi / 3.0)
        nz = np.cos(theta)[:, None] * noise[:, 0][None, :] + np.sin(theta)[:, None] * noise[:, 1][None, :]
        c = math.exp(-2.0 * math.pi * min(sr * 0.45, float(np.mean(f1)) * TUBE_NOISE_HARM) / sr)
        nz, self.noise_zi = lfilter([1.0 - c], [1.0, -c], nz, axis=1, zi=self.noise_zi)
        nz *= TUBE_NOISE / NOISE_RMS

        t = np.arange(frames) / frames
        press = p0 + (p1 - p0) * t                      # (frames,)
        gain = TUBE_G0 + TUBE_G1 * press
        out = np.empty((self.ranks, frames))
        self.pos = _loop(self.buf, self.pos, self.zi, d0, d1, b, a, gain, press,
                         np.ascontiguousarray(nz), self.bias, out)
        # Radiated sound: the first difference, scaled to keep the level of the
        # mode that speaks (the note, or the octave when overblown).
        rad = np.diff(out, axis=1, prepend=self.last[:, None])
        self.last = out[:, -1].copy()
        f_mode = np.minimum(f1 * max(1.0, mode), sr * 0.25)
        return rad / (2.0 * np.sin(math.pi * f_mode / sr))[:, None]


if __name__ == "__main__":
    # Offline check (no audio): at several notes and pressures, the pitch, how
    # much of the sound is tone, the level; then the time per block. It fails
    # if the pipe is out of tune, does not catch, or does not overblow.
    import time

    SR = 48000
    BLOCK = 512

    def pink(n, seed=0):
        from .engine import PINK_DIRECT, PINK_GAINS, PINK_POLES, PINK_SCALE
        w = np.random.default_rng(seed).standard_normal((n, 2)) * 0.4
        src = w * PINK_DIRECT
        for pole, g in zip(PINK_POLES, PINK_GAINS):
            src = src + lfilter([g], [1.0, -pole], w, axis=0)
        return src * PINK_SCALE

    def run(f, pressure, seconds=2.0, mode=1.0):
        tube = Tube(SR, 1, np.random.default_rng(1))
        noise = pink(int(seconds * SR))
        out = [tube.process(np.array([f]), pressure, pressure, noise[i * BLOCK:(i + 1) * BLOCK], mode)[0]
               for i in range(int(seconds * SR / BLOCK))]
        return np.concatenate(out)

    def analyse(y, f):
        """(pitch error in cents of the strongest peak near f, tone/noise dB, rms)"""
        y = y[len(y) // 2:]
        y = y - y.mean()
        assert np.all(np.isfinite(y)), "NaN or inf"
        Y = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 18)) ** 2
        fr = np.fft.rfftfreq(1 << 18, 1.0 / SR)
        m = (fr > f * 0.7) & (fr < f * 1.4)
        pk = fr[m][np.argmax(Y[m])]
        harm = sum(Y[np.abs(fr - k * pk) < 4.0].sum() for k in range(1, 16))
        hnr = 10.0 * math.log10(harm / max(Y.sum() - harm, 1e-30) + 1e-30)
        return 1200.0 * math.log2(pk / f), hnr, float(np.sqrt(np.mean(y ** 2)))

    bad = []
    for f in (65.4, 130.8, 523.3, 1046.5):
        rows = []
        for pr in (0.3, 0.6, 0.8, 1.0, 1.3):
            cents, hnr, rms = analyse(run(f, pr), f)
            rows.append((pr, cents, hnr, rms))
            print(f"f {f:7.1f}  pressure {pr:.1f}: pitch {cents:+4.0f} cents  tone/noise {hnr:+5.1f} dB  rms {rms:.3f}")
            if pr >= 0.8 and (abs(cents) > 15.0 or hnr < 3.0):
                bad.append(f"{f} Hz at pressure {pr}: out of tune or not singing")
        if rows[3][3] < 3.0 * rows[0][3]:
            bad.append(f"{f} Hz: singing is not much louder than breathing")
        cents, hnr, rms = analyse(run(f, 1.0, mode=2.0), 2.0 * f)
        print(f"f {f:7.1f}  overblown:     pitch {cents:+4.0f} cents from the octave  tone/noise {hnr:+5.1f} dB  rms {rms:.3f}")
        if abs(cents) > 15.0 or rms > 2.0 * rows[3][3]:
            bad.append(f"{f} Hz: overblow is not the octave, or too loud")

    tube = Tube(SR, 3, np.random.default_rng(1))
    noise = pink(BLOCK * 200)
    for f in (130.8, 1046.5):
        t0 = time.perf_counter()
        for i in range(200):
            tube.process(np.full(3, f), 0.8, 0.8, noise[i * BLOCK:(i + 1) * BLOCK])
        print(f"time per block, 3 ranks at {f:.0f} Hz: {(time.perf_counter() - t0) / 200 * 1e3:.2f} ms")
    print("\n".join(bad) if bad else "all good")
    raise SystemExit(1 if bad else 0)
