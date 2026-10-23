"""Stereo Feedback Delay Network (FDN) reverb on the synth output bus.

A wet/dry reverb applied to the synth's stereo signal after saturation. It adds
NO latency to the dry path: the dry signal passes straight through the callback
and the reverb only adds a decaying tail on top. process() returns the wet signal
only, so a host can feed other sources (a sung voice, say) into its own instance.

Why an FDN and not the old Freeverb (8 combs + 4 allpass)? On a SUSTAINED tone --
the wind/organ drone is the worst case -- Freeverb's sparse, static comb echoes
ring as a metallic flutter, and it jumps straight to a diffuse-ish tail with no
sense of a large stone volume. An FDN's 16 delay lines, mixed every block by an
orthonormal Hadamard matrix, scatter energy among all lines so echo density grows
quadratically into a smooth tail almost immediately -- the lush "church" wash.

Crucially the network is LINEAR AND TIME-INVARIANT (LTI), exactly like a real
stone room: a pure sustained tone in -> the same tone plus its reverberation AT
THE SAME FREQUENCY out, never an inharmonic "fizz" of new frequencies. That is
why the delay reads are plain integer reads here -- an earlier version chorus-
MODULATED them to smear residual ringing, but a time-varying delay generates
sidebands, which on the wind/organ's tonal material is heard as a sizzling halo.
A real church does not chorus, so neither do we; density comes from the matrix,
not from movement. (If a future version wants air movement, modulate only very
shallowly and only the tail, never enough to detune the held note audibly.)

Efficiency note (same trick as the old Freeverb, why not a plain lfilter): the
feedback matrix couples all lines every sample, but it mixes the delay OUTPUTS,
which are reads from buffers at least as long as the processing sub-block. If we
split each block into sub-blocks no longer than the shortest delay, no sample
within a sub-block ever feeds back on itself, so the whole sub-block vectorizes:
per sub-block we do 16 gathers, 16 order-1 damping lfilters, one (sub x 16)@(16 x
16) matmul, and 16 scatters. The shortest line is ~21 ms (>1000 samples at 48
kHz), so a 512-sample block is a single sub-block in practice -- cost on par with
the old 8-comb/4-allpass Freeverb.

The public surface is unchanged: StereoReverb(sr) and process(dry, room, damping)
-> wet (frames, 2). room 0..1 sets the decay length (RT60), damping 0..1 the
high-frequency darkness of the tail, exactly as before.
"""
import math

import numpy as np
from scipy.signal import lfilter

_N = 16  # number of delay lines (a power of two so the Hadamard matrix is exact)

# Delay-line lengths in samples at 48 kHz: primes spread ~21-53 ms so the modal
# series is dense and mutually irregular (no common divisor -> no flutter pitch).
_DELAYS_48K = [1031, 1123, 1229, 1321, 1429, 1523, 1619, 1733,
               1847, 1933, 2027, 2137, 2237, 2339, 2441, 2543]
_REF_SR = 48000.0

_PREDELAY_MS = 18.0          # gap before the diffuse tail -> sense of a large volume
# Early reflections: (time_ms, gain_L, gain_R) -- a rough image-source pattern for a
# nave (a few strong first bounces off the side walls/floor before the diffuse tail).
_ER = [(11, .90, .60), (17, .60, .85), (23, .80, .50), (29, .50, .75),
       (37, .65, .55), (43, .45, .65), (53, .55, .40), (61, .40, .55)]
_ER_GAIN = 0.45             # early-reflection level relative to the diffuse tail
_TAIL_GAIN = 1.0
# The pre-delay/early-reflection ring buffer must be longer than the deepest tap
# PLUS one processing block, or the deepest taps read samples this block just
# overwrote -> a per-block delay jump that sprays sidebands at the block rate. This
# headroom comfortably covers any PortAudio block size we will ever be handed.
_ER_HEADROOM = 8192

# room 0..1 -> RT60 seconds (geometric): 0.5 s small chapel .. cathedral. The
# damping lowpass shortens the broadband decay below this nominal figure (the highs
# die faster than the lows -- the frequency-dependent decay of a real stone room),
# so the knob top is set well past 6 s to actually reach a cathedral tail.
_RT60_MIN, _RT60_MAX = 0.5, 8.0
_DAMP_MAX = 0.72            # damping 0..1 -> feedback lowpass coef 0.._DAMP_MAX

# Bass bloom: a low-shelf in the feedback that makes the low end ring LONGER than
# the broadband tail -- the warm low-mid bloom of a real stone room (the damping
# lowpass already darkens the highs; this lifts the lows the other way). Below
# _SHELF_FC the per-line decay gain g is raised to g**(1/(1+k)): since g < 1 the
# result is a higher gain that is still < 1 (so the loop stays stable at every
# frequency, LTI), and the low-band RT60 becomes exactly (1+k)x the broadband RT60.
# bass 0 -> k=0 -> the shelf adds exactly nothing (identical to the pre-bloom reverb).
_SHELF_FC = 300.0          # crossover: lows below this get the longer decay
_BASS_MAX = 0.6            # bass 1.0 -> low-band RT60 up to +60% longer than broadband

_IN_GAIN = 0.18            # input injected into every line
_WET_MAKEUP = 1.7         # so a fully-wet mix sits near the dry's loudness (tuned)


def _hadamard(n: int) -> np.ndarray:
    """Sylvester-construction Hadamard matrix, normalised to be orthonormal
    (lossless feedback: the matrix neither adds nor removes energy, so the per-line
    decay gains alone set the RT60 and the loop is unconditionally stable)."""
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


class StereoReverb:
    """16-line stereo FDN. process() returns the WET signal only (no dry mixed in);
    the caller does the wet/dry blend so the same bus can serve several dry sources."""

    def __init__(self, sr: int):
        self.sr = sr
        scale = sr / _REF_SR

        def sc(d: int) -> int:
            return max(1, int(round(d * scale)))

        self.delays = [sc(d) for d in _DELAYS_48K]
        self.bufs = [np.zeros(d) for d in self.delays]
        self.pos = [0] * _N
        self.damp_zi = [np.zeros(1) for _ in range(_N)]
        # Per-line crossover lowpass state for the bass-bloom shelf (see process()).
        self.shelf_zi = [np.zeros(1) for _ in range(_N)]
        self.shelf_c = math.exp(-2.0 * math.pi * _SHELF_FC / sr)  # one-pole LP coef
        self.H = _hadamard(_N)

        # Output decorrelation: two distinct Hadamard sign rows tap the line outputs
        # into L and R, so the two channels are uncorrelated (a wide diffuse field).
        raw = _hadamard(_N) * np.sqrt(_N)          # back to +-1 entries
        self.cL = raw[3].copy()
        self.cR = raw[7].copy()

        # Pre-delay + early-reflection source buffer (mono, feed-forward: a plain tap
        # delay, so it never constrains the sub-block size below). Length includes a
        # full-block headroom past the deepest tap (see _ER_HEADROOM).
        self.pre_d = max(2, int(round(_PREDELAY_MS * 1e-3 * sr)))
        self.er_taps = [(max(1, int(round(t * 1e-3 * sr))), gl, gr) for t, gl, gr in _ER]
        er_max = max(self.pre_d, max(d for d, _, _ in self.er_taps))
        self.pre_len = er_max + _ER_HEADROOM
        self.pre = np.zeros(self.pre_len)
        self.pre_pos = 0

        # chunk no longer than the shortest delay, so no line feeds back on itself
        # inside a sub-block (lets the whole sub-block vectorize).
        self.max_sub = max(1, min(self.delays) - 1)

    def process(self, dry: np.ndarray, room: float, damping: float,
                bass: float = 0.0) -> np.ndarray:
        """dry: (frames, 2). Returns wet (frames, 2). bass 0..1 lengthens the
        low-frequency tail (warm bloom); 0 = neutral (identical to no bloom)."""
        frames = dry.shape[0]
        room = float(np.clip(room, 0.0, 1.0))
        damping = float(np.clip(damping, 0.0, 1.0))
        bass = float(np.clip(bass, 0.0, 1.0))

        rt60 = _RT60_MIN * (_RT60_MAX / _RT60_MIN) ** room
        delays = np.asarray(self.delays, dtype=np.float64)
        g = 10.0 ** (-3.0 * delays / (rt60 * self.sr))   # Jot per-line decay gains
        damp1 = _DAMP_MAX * damping
        # Per-line bass-bloom shelf boost: raise the low-frequency decay gain from g to
        # g**(1/kbass) (kbass >= 1), so the low-band RT60 is kbass x the broadband one.
        # g < 1 -> g**(1/kbass) is larger but still < 1, so the loop stays stable with no
        # hard cap. bj is the shelf's DC boost relative to the broadband gain g.
        kbass = 1.0 + _BASS_MAX * bass
        bj = g ** (1.0 / kbass - 1.0) if bass > 0.0 else None  # per-line DC boost (>= 1)
        shelf_c = self.shelf_c

        # --- pre-delay + early reflections (feed-forward, whole block at once) ---
        mono = (dry[:, 0] + dry[:, 1]) * 0.5
        pre, ppos, plen = self.pre, self.pre_pos, self.pre_len
        k = np.arange(frames)
        pre[(ppos + k) % plen] = mono
        self.pre_pos = (ppos + frames) % plen
        x = pre[(ppos + k - self.pre_d) % plen]          # pre-delayed input to the FDN
        er_l = np.zeros(frames)
        er_r = np.zeros(frames)
        for d, gl, gr in self.er_taps:
            tap = pre[(ppos + k - d) % plen]
            er_l += gl * tap
            er_r += gr * tap

        # --- diffuse tail (the FDN) ---
        out_l = np.empty(frames)
        out_r = np.empty(frames)
        start = 0
        while start < frames:
            end = min(start + self.max_sub, frames)
            m = end - start
            xs = x[start:end]
            n = np.arange(m)

            douts = np.empty((m, _N))    # delay-line outputs (the output tap)
            damped = np.empty((m, _N))   # damped + decayed (fed to the matrix)
            for j in range(_N):
                buf = self.bufs[j]
                D = buf.shape[0]
                o = buf[(self.pos[j] + n) % D]
                douts[:, j] = o
                ds, self.damp_zi[j] = lfilter([1.0 - damp1], [1.0, -damp1], o,
                                              zi=self.damp_zi[j])
                if bj is not None:
                    # low-shelf as a parallel add: ds + (Bj-1) * lowpass(ds). Unity at
                    # HF (shelf vanishes), boost Bj at DC -> lows decay slower. Zero at
                    # bass=0 since Bj=1, so this is bit-identical to the plain tail.
                    lo, self.shelf_zi[j] = lfilter([1.0 - shelf_c], [1.0, -shelf_c], ds,
                                                   zi=self.shelf_zi[j])
                    ds = ds + (bj[j] - 1.0) * lo
                damped[:, j] = ds * g[j]

            fb = damped @ self.H                          # orthonormal energy scatter
            for j in range(_N):
                buf = self.bufs[j]
                D = buf.shape[0]
                pos = self.pos[j]
                buf[(pos + n) % D] = xs * _IN_GAIN + fb[:, j]
                self.pos[j] = (pos + m) % D

            out_l[start:end] = (douts @ self.cL) / np.sqrt(_N)
            out_r[start:end] = (douts @ self.cR) / np.sqrt(_N)
            start = end

        wet_l = _TAIL_GAIN * out_l + _ER_GAIN * er_l
        wet_r = _TAIL_GAIN * out_r + _ER_GAIN * er_r
        return np.stack([wet_l, wet_r], axis=1) * _WET_MAKEUP


if __name__ == "__main__":
    # Offline smoke test (no audio hardware), like feedback.py. Verifies: a pure tone
    # stays a pure tone (LTI -- no inharmonic "fizz"), an impulse decays to an RT60
    # that tracks the room knob, the tail is stable, and L/R are decorrelated.
    SR = 48000
    BLOCK = 512

    def render_sine(freq, room, damping, secs=3.0):
        rv = StereoReverb(SR)
        ph = 0.0
        out = []
        for _ in range(int(secs * SR / BLOCK)):
            n = np.arange(BLOCK)
            t = (ph + n) / SR
            ph += BLOCK
            s = 0.3 * np.sin(2 * np.pi * freq * t)
            out.append(rv.process(np.stack([s, s], axis=1), room, damping))
        return np.concatenate(out, axis=0)

    print("pure-tone fidelity (% energy NOT within +-3 Hz of the input -- want ~0):")
    for f0 in (220.0, 440.0, 880.0):
        y = render_sine(f0, 0.7, 0.75)[SR:, 0]
        Y = np.abs(np.fft.rfft(y * np.hanning(y.size)))
        f = np.fft.rfftfreq(y.size, 1.0 / SR)
        spray = 100.0 * (1.0 - (Y[np.abs(f - f0) <= 3.0] ** 2).sum() / (Y ** 2).sum())
        print(f"  {f0:6.0f} Hz -> spray = {spray:5.2f}%")

    def run_impulse(room, damping, secs=8.0):
        rv = StereoReverb(SR)
        imp = np.zeros((BLOCK, 2))
        imp[0] = 1.0
        out = []
        for b in range(int(secs * SR / BLOCK)):
            out.append(rv.process(imp if b == 0 else np.zeros((BLOCK, 2)), room, damping))
        return np.concatenate(out, axis=0)

    def rt60_of(sig):
        env = np.maximum(np.abs(sig[:, 0]), np.abs(sig[:, 1]))
        edc = np.cumsum(env[::-1] ** 2)[::-1]
        edc = 10.0 * np.log10(edc / edc[0] + 1e-20)
        i5 = int(np.argmax(edc <= -5.0))
        i35 = int(np.argmax(edc <= -35.0))
        if i35 <= i5:
            return float("nan")
        slope = (edc[i35] - edc[i5]) / ((i35 - i5) / SR)
        return -60.0 / slope

    print("RT60 vs room knob (damping=0.3):")
    for room in (0.0, 0.3, 0.7, 1.0):
        print(f"  room={room:.1f} -> RT60 ~ {rt60_of(run_impulse(room, 0.3)):.2f} s")

    rv = StereoReverb(SR)
    rng = np.random.default_rng(0)
    peak = rms_in = rms_wet = 0.0
    nb = int(4.0 * SR / BLOCK)
    for _ in range(nb):
        dry = 0.2 * rng.standard_normal((BLOCK, 2))
        wet = rv.process(dry, 0.7, 0.4)
        peak = max(peak, float(np.abs(wet).max()))
        rms_in += float(np.mean(dry ** 2))
        rms_wet += float(np.mean(wet ** 2))
    rms_in = np.sqrt(rms_in / nb)
    rms_wet = np.sqrt(rms_wet / nb)
    print(f"sustained noise: peak={peak:.2f} (stable={peak < 4.0}), "
          f"wet/dry RMS ratio={rms_wet / rms_in:.2f}")

    sig = run_impulse(0.7, 0.3)
    tail = sig[SR // 2:]
    corr = float(np.corrcoef(tail[:, 0], tail[:, 1])[0, 1])
    print(f"L/R correlation in the tail: {corr:+.2f} (want |.| small)")
