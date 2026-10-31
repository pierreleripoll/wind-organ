"""A blown pipe: the wind finds a tube and it starts to sing.

This is the "organic" end of organ_organic. The clean organ is a sum of sines.
Here the sound is made by the air itself, like wind in a crack of a cave or in
a ruin that happens to sound like an organ pipe.

The model is a flue pipe (an organ flue pipe, a flute, or wind across a crack),
one loop per rank of the pipe:

                       wind noise ---------+
                           v               v
      +--> [jet delay] --> [jet: tanh] --> (+) --> air column (one period) --+--> out
      |                                     ^                               |
      |                                     +--- reflection <--+            |
      |                                                        |            |
      +--------------------------------------------------------+-- [wall losses]

- The air column is a delay one period long. The sound goes down the pipe and
  comes back a bit weaker (the reflection, the wall losses), so alone it rings
  at the note and at all its harmonics, then dies.
- The jet is the thin sheet of air from the slit to the edge. It moves with
  the air at the mouth, but late: the air needs time to travel along the jet.
  Then it hits the edge on one side or the other (tanh) and pushes the column.
- That travel time does everything. The jet pushes a mode in step only when it
  arrives about half a period of this mode late. More pressure = a faster jet =
  a shorter travel. So:
  - Too little pressure: the jet is slow and weak, it pushes nothing in step.
    The pipe only breathes: the wind noise rings in the column, a faint pitched
    "hoo". (The organ's "air" layer in engine.py makes most of this breath.)
  - Normal pressure: the jet fits half a period of the note. The pipe sings.
    Its pitch rises a little with the pressure, like a real flute.
  - A strong gust: the travel now fits half a period of the octave better, and
    the pipe overblows by itself. It stays up there until the pressure falls
    well below (hysteresis, a real flute does the same). Low pipes are slow:
    they speak late and overblow later.
- The jet sends back the opposite of what it reads (the edge is on the other
  side). With a half period of travel, this damps the even modes: the tone is
  mostly odd harmonics, like a stopped organ pipe (a bourdon). The jet hits the
  edge a bit off centre (a random offset per pipe), which adds a little of the
  even ones back.
- The wind noise moves the jet and shakes the column all the time (turbulence
  at the mouth), so even the sung tone breathes and wavers, the way a real pipe
  in moving air does. It is lowpassed at a few
  harmonics of the note first: the breath of the pipe is a deep "hoo" around
  the note. The broad hiss is already in the wind layer.
- The sound out is the change of the air in the tube (its first difference),
  like the sound a pipe mouth radiates: it lifts the harmonics by 6 dB per
  octave. It is scaled to keep the level of the mode that speaks. To know this
  mode, each block compares the column with itself half a period back: the same
  sign means it is on the octave.

The loop has one recursion per sample, which is very slow in plain Python. So
it is compiled with numba (_loop below): it runs sample by sample like C. It is
compiled when this module is imported (and cached on disk), never in the audio
callback.
"""
import math

import numba
import numpy as np
from scipy.signal import lfilter

TUBE_BUF = 4096           # delay buffer per rank (a power of two): pipes down to ~15 Hz
TUBE_F_MIN = 15.0
NOISE_RMS = 0.132         # RMS of the engine's pink wind noise, to bring it to 1
TUBE_REFLECT = 0.985      # the column keeps this much per round trip: how long it rings
TUBE_LOSS_HARM = 8.0      # wall losses: a lowpass at this harmonic of the note
TUBE_DC_HARM = 0.02       # a DC blocker in the loop, at this fraction of the note
TUBE_JET_GAIN = 0.12      # how hard the jet pushes. More = it sings at lower pressure,
                          # but its pitch moves more with the pressure
TUBE_JET_HALF = 0.36      # pressure where the jet travel is half a period: in tune.
                          # It sings from ~0.2, overblows ~1.15, comes back ~0.8.
TUBE_JET_P_MIN = 0.05     # (the jet travel stops growing below this pressure)
TUBE_TUNE_CENTS = 4.0     # the bore is this much longer, so the pipe is in tune
                          # around pressure 0.5 and not at TUBE_JET_HALF
TUBE_NOISE = 0.15         # wind noise on the jet
TUBE_BREATH = 1.5         # wind noise straight into the column (turbulence at the mouth)
TUBE_NOISE_HARM = 4.0     # the noise is lowpassed at this harmonic of the note
TUBE_BIAS = (0.3, 0.5)    # offset of the jet (random per pipe)
TUBE_OUT = 0.024          # output level
TUBE_MODE_TAU = 0.05      # the output level follows a jump to the octave this fast (s)


# The signature makes numba compile at import time, not at the first call (that
# would block the audio for a second).
@numba.njit("int64(float64[:, ::1], float64[:, ::1], float64[:, ::1], int64, float64[::1],"
            " float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64,"
            " float64, float64[:, ::1], float64, float64[:, ::1], float64[:, ::1])",
            cache=True, nogil=True)
def _loop(buf, jbuf, st, pos, d0, d1, t0, t1, c, adc, p0, p1, nz, bias, out, corr):
    """The pipe, sample by sample, for each rank.

    buf: (ranks, TUBE_BUF) the air column. jbuf: the same size, what the column
    did at the mouth (the jet reads it late). st: (ranks, 3) filter states.
    pos: write position of the first sample. d0, d1: delay of the column at the
    start and the end of the block. t0, t1: the period, the same. c, adc: poles
    of the wall losses and of the DC blocker. p0, p1: pressure at the start and
    the end. nz: (ranks, frames) wind noise. out: (ranks, frames) is filled with
    the column at the mouth. corr: (ranks, 2) gets sum(s * s half a period
    back) and sum(s * s), to see if the pipe overblows. Returns the next write
    position."""
    ranks, frames = nz.shape
    mask = buf.shape[1] - 1
    tb = math.tanh(bias)
    k = -TUBE_JET_GAIN / (1.0 - tb * tb)   # small-signal gain of the jet: TUBE_JET_GAIN
    for r in range(ranks):
        lp, x1, y1 = st[r, 0], st[r, 1], st[r, 2]
        cr, ar = c[r], adc[r]
        acc, en = 0.0, 0.0
        for n in range(frames):
            t = n / frames
            w = pos + n
            p = p0 + (p1 - p0) * t
            per = t0[r] + (t1[r] - t0[r]) * t
            # The air column, one round trip back, then the wall losses.
            rp = w - (d0[r] + (d1[r] - d0[r]) * t)
            i0 = int(math.floor(rp))
            fr = rp - i0
            a = buf[r, i0 & mask]
            yb = a + (buf[r, (i0 + 1) & mask] - a) * fr
            lp = (1.0 - cr) * yb + cr * lp
            y = lp - x1 + ar * y1
            x1, y1 = lp, y
            # The jet: what the mouth did one jet travel ago (shorter when the
            # pressure is higher), inverted, through the edge (tanh). Its speed
            # and its flow grow with the square root of the pressure.
            dj = min(max(0.5 * math.sqrt(TUBE_JET_HALF / max(p, TUBE_JET_P_MIN)) * per, 1.5), mask - 3.0)
            rp = w - dj
            i0 = int(math.floor(rp))
            fr = rp - i0
            a = jbuf[r, i0 & mask]
            jin = a + (jbuf[r, (i0 + 1) & mask] - a) * fr
            sq = math.sqrt(max(p, 0.0))
            jet = sq * (math.tanh(k * jin + nz[r, n] + bias) - tb)
            s = TUBE_REFLECT * y + jet + TUBE_BREATH * sq * nz[r, n]
            buf[r, w & mask] = s
            jbuf[r, w & mask] = y
            out[r, n] = s
            # Overblow detector: the column half a period back.
            rp = w - 0.5 * per
            i0 = int(math.floor(rp))
            fr = rp - i0
            a = buf[r, i0 & mask]
            acc += s * (a + (buf[r, (i0 + 1) & mask] - a) * fr)
            en += s * s
        st[r, 0], st[r, 1], st[r, 2] = lp, x1, y1
        corr[r, 0], corr[r, 1] = acc, en
    return (pos + frames) & mask


def _loop_delay(f: np.ndarray, c: np.ndarray, adc: np.ndarray, sr: int) -> np.ndarray:
    """Delay of the column (samples) for the notes f: one period (a bit longer,
    see TUBE_TUNE_CENTS) minus the delay of the wall losses and the DC blocker
    at the note, so the loop is in tune."""
    w = 2.0 * np.pi * f / sr
    z = np.exp(-1j * w)
    h = (1.0 - c) / (1.0 - c * z) * (1.0 - z) / (1.0 - adc * z)
    return sr / f * 2.0 ** (TUBE_TUNE_CENTS / 1200.0) + np.angle(h) / w


class Tube:
    """One blown pipe with `ranks` loops (its detuned copies).

    process() takes the frequency of each rank, the pressure at the start and
    the end of the block, and the wind noise. It returns the sound of each
    rank, (ranks, frames), roughly in -1..1."""

    def __init__(self, sr: int, ranks: int, rng: np.random.Generator):
        self.sr = sr
        self.rng = rng
        self.bias = rng.uniform(*TUBE_BIAS)   # each pipe (each crack) is a bit different
        self.ranks = 0
        self.buf = np.zeros((0, TUBE_BUF))
        self.jbuf = np.zeros((0, TUBE_BUF))
        self.st = np.zeros((0, 3))
        self.noise_zi = np.zeros((0, 1))
        self.last = np.zeros(0)   # last sample of each rank (for the radiated output)
        self.octave = np.zeros(0)  # 0 = on the note, 1 = overblown to the octave
        self.out_gain = None       # output gain of the last block, per rank
        self.pos = 0
        self.f_prev = None
        self.set_ranks(ranks)

    def set_ranks(self, ranks: int) -> None:
        """Keep the loops that stay. New loops start silent."""
        if ranks == self.ranks:
            return
        k = min(ranks, self.ranks)

        def resize(a: np.ndarray, shape: tuple) -> np.ndarray:
            b = np.zeros(shape)
            b[:k] = a[:k]
            return b

        self.buf = resize(self.buf, (ranks, TUBE_BUF))
        self.jbuf = resize(self.jbuf, (ranks, TUBE_BUF))
        self.st = resize(self.st, (ranks, 3))
        self.noise_zi = resize(self.noise_zi, (ranks, 1))
        self.last = resize(self.last, (ranks,))
        self.octave = resize(self.octave, (ranks,))
        self.ranks = ranks
        self.out_gain = None
        self.f_prev = None

    def process(self, freqs: np.ndarray, p0: float, p1: float, noise: np.ndarray) -> np.ndarray:
        """freqs: (ranks,) frequency of each rank at the end of the block (it
        glides from the last block's). p0, p1: air pressure at the start and
        the end of the block. noise: (frames, 2) wind noise. Returns (ranks, frames)."""
        sr = self.sr
        frames = noise.shape[0]
        f1 = np.clip(np.asarray(freqs, dtype=np.float64), TUBE_F_MIN, sr * 0.45)
        f0 = f1 if self.f_prev is None else self.f_prev
        self.f_prev = f1
        c = np.exp(-2.0 * np.pi * np.minimum(sr * 0.45, f1 * TUBE_LOSS_HARM) / sr)
        adc = np.exp(-2.0 * np.pi * f1 * TUBE_DC_HARM / sr)

        # Each rank breathes its own blend of the left and right wind noise,
        # lowpassed around the first harmonics of the note.
        theta = np.arange(self.ranks) * (math.pi / 3.0)
        nz = np.cos(theta)[:, None] * noise[:, 0][None, :] + np.sin(theta)[:, None] * noise[:, 1][None, :]
        cn = math.exp(-2.0 * math.pi * min(sr * 0.45, float(np.mean(f1)) * TUBE_NOISE_HARM) / sr)
        nz, self.noise_zi = lfilter([1.0 - cn], [1.0, -cn], nz, axis=1, zi=self.noise_zi)
        nz = np.ascontiguousarray(nz * (TUBE_NOISE / NOISE_RMS))

        out = np.empty((self.ranks, frames))
        corr = np.zeros((self.ranks, 2))
        self.pos = _loop(self.buf, self.jbuf, self.st, self.pos,
                         _loop_delay(f0, c, adc, sr), _loop_delay(f1, c, adc, sr),
                         sr / f0, sr / f1, c, adc, float(p0), float(p1), nz, self.bias, out, corr)

        # On the octave, the column looks the same half a period back.
        same = corr[:, 0] / (corr[:, 1] + 1e-12)
        target = np.clip((same + 0.2) / 0.8, 0.0, 1.0)
        self.octave += (target - self.octave) * (1.0 - math.exp(-frames / (sr * TUBE_MODE_TAU)))
        # Radiated sound: the first difference, scaled to keep the level of the
        # mode that speaks. Ramped across the block.
        f_mode = np.minimum(f1 * (1.0 + self.octave), sr * 0.25)
        gain = TUBE_OUT / (2.0 * np.sin(math.pi * f_mode / sr))
        g0 = gain if self.out_gain is None else self.out_gain
        self.out_gain = gain
        rad = np.diff(out, axis=1, prepend=self.last[:, None])
        self.last = out[:, -1].copy()
        return rad * np.linspace(g0, gain, frames, axis=1)


if __name__ == "__main__":
    # Offline check (no audio): at several notes and pressures, the pitch, how
    # much of the sound is tone, the level; a gust that overblows and comes
    # back; then the time per block. It fails if the pipe is out of tune, does
    # not sing, does not overblow, or does not come back.
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

    def run(f, pressure):
        """pressure: one value per block."""
        tube = Tube(SR, 1, np.random.default_rng(1))
        noise = pink(len(pressure) * BLOCK)
        prev = pressure[0]
        out = []
        for i, pr in enumerate(pressure):
            out.append(tube.process(np.array([f]), prev, pr, noise[i * BLOCK:(i + 1) * BLOCK])[0])
            prev = pr
        return np.concatenate(out)

    def analyse(y, f):
        """(pitch error in cents of the strongest peak near f, tone/noise dB, rms)"""
        y = y - y.mean()
        assert np.all(np.isfinite(y)), "NaN or inf"
        Y = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 18)) ** 2
        fr = np.fft.rfftfreq(1 << 18, 1.0 / SR)
        m = (fr > f * 0.7) & (fr < f * 1.4)
        pk = fr[m][np.argmax(Y[m])]
        harm = sum(Y[np.abs(fr - k * pk) < 4.0].sum() for k in range(1, 16))
        hnr = 10.0 * math.log10(harm / max(Y.sum() - harm, 1e-30) + 1e-30)
        return 1200.0 * math.log2(pk / f), hnr, float(np.sqrt(np.mean(y ** 2)))

    def blocks(seconds):
        return int(seconds * SR / BLOCK)

    bad = []
    for f in (65.4, 130.8, 523.3, 1046.5, 1975.5):
        # Low pipes speak slowly: hold each pressure 3 s, look at the last 1.5 s.
        rms = {}
        for pr in (0.1, 0.3, 0.5, 0.8):
            y = run(f, np.full(blocks(3.0), pr))[-int(1.5 * SR):]
            cents, hnr, rms[pr] = analyse(y, f)
            print(f"f {f:7.1f}  pressure {pr:.1f}: pitch {cents:+4.0f} cents  tone/noise {hnr:+5.1f} dB  rms {rms[pr]:.3f}")
            if pr >= 0.3 and (abs(cents) > 15.0 or hnr < 10.0):
                bad.append(f"{f} Hz at pressure {pr}: out of tune or not singing")
        if rms[0.5] < 3.0 * rms[0.1]:
            bad.append(f"{f} Hz: singing is not much louder than breathing")
        # A gust: 0.6 for 2 s, up to 1.5 over 2 s and hold 2 s, back to 0.5 and hold 3 s.
        gust = np.concatenate([np.full(blocks(2.0), 0.6), np.linspace(0.6, 1.5, blocks(2.0)),
                               np.full(blocks(2.0), 1.5), np.linspace(1.5, 0.5, blocks(1.0)),
                               np.full(blocks(3.0), 0.5)])
        y = run(f, gust)
        cents_up, hnr_up, rms_up = analyse(y[int(5.0 * SR):int(6.0 * SR)], 2.0 * f)
        cents_back, hnr_back, _ = analyse(y[int(8.5 * SR):], f)
        print(f"f {f:7.1f}  gust: overblown {cents_up:+4.0f} cents from the octave, tone/noise {hnr_up:+5.1f} dB,"
              f" rms {rms_up:.3f}; back {cents_back:+4.0f} cents")
        if abs(cents_up) > 20.0 or hnr_up < 10.0 or rms_up > 2.0 * rms[0.5]:
            bad.append(f"{f} Hz: the gust does not overblow to the octave, or it is too loud")
        if abs(cents_back) > 15.0 or hnr_back < 10.0:
            bad.append(f"{f} Hz: the pipe does not come back to the note after the gust")

    tube = Tube(SR, 3, np.random.default_rng(1))
    noise = pink(BLOCK * 200)
    for f in (130.8, 1046.5):
        t0 = time.perf_counter()
        for i in range(200):
            tube.process(np.full(3, f), 0.6, 0.6, noise[i * BLOCK:(i + 1) * BLOCK])
        print(f"time per block, 3 ranks at {f:.0f} Hz: {(time.perf_counter() - t0) / 200 * 1e3:.2f} ms")
    print("\n".join(bad) if bad else "all good")
    raise SystemExit(1 if bad else 0)
