"""The wind + organ synth engine. Pure DSP: no audio device, no threads, no input.

Signal flow per block:
    pink noise (Paul Kellet 6-pole IIR), two decorrelated columns (L/R) ->
        either three parallel RBJ bandpass biquads with per-band gains (the
        "spectral tilt") or one bandpass + one lowpass (single-band)
    -> multiply by level x gust LFO (+ optional buffet flutter, gust sweep L/R)
    -> optional bourdon voices (narrow bandpass at root + 5th + 3rd)
    -> optional wind chord (narrow bandpasses at the organ_chord notes)
    -> optional edge whistles (bandpasses that sing only in strong wind)
    -> optional organ drone (additive pipe partials + pitched breath; with
       organ_organic, less clean partials and blown tubes, see tube.py)
    -> tanh saturation (drive)                                   = voice()
    -> stereo FDN reverb, wind_gain                              = room()
    -> subsonic high-pass + brick-wall limiter (speaker safety)  = protect()

The host gives two controls per block: a pitch in Hz (FREQ_LO..FREQ_HI, see
pitch_to_freq) and a level in 0..1. It does its own smoothing and gating (see
controls.py for a simple one). render() runs the full chain. A host that needs
to insert something (a mic before the limiter, say) calls the three steps itself.

The organ layers a pipe drone on top of the wind (it does not replace it). It is
driven by the SAME gust LFO as the wind, which swells, brightens and sharpens it
with the audible wind, so it sounds like wind resonating a building, not a
separate instrument. In hold mode the pipe keeps one fixed note; the waver,
vibrato and gust overblow still detune and swell it, so it keeps breathing.
With organ_chord it plays several pipes at just-intonation ratios over that
root, each pipe as organ_ranks copies that beat slowly, spread in stereo.

Filter coefficients are rebuilt per block (cheap), so knob changes apply within
one block.
"""
import math

import numpy as np
from scipy.signal import butter, iirfilter, lfilter, sosfilt

from .chords import parse_chord
from .params import Params
from .reverb import StereoReverb
from .tube import Tube

SR = 48000
FREQ_LO, FREQ_HI = 150.0, 5000.0   # pitch range the wind maps onto (Hz, log)
RUMBLE_CUTOFF = 90.0               # single-band mode: lowpass rumble under the bandpass

# Paul Kellet's refined pink noise filter (6 parallel one-poles + white passthrough).
# Sounds smoother than Voss-McCartney; cheap when run via lfilter per pole.
PINK_POLES = [0.99886, 0.99332, 0.96900, 0.86650, 0.55000, -0.7616]
PINK_GAINS = [0.0555179, 0.0750759, 0.1538520, 0.3104856, 0.5329522, -0.0168980]
PINK_DIRECT = 0.5362
PINK_SCALE = 0.11

# Storm morph shape (the user dial is Params.storm). m = storm * level^1.5 grows
# near full level; soft = storm * (1 - level) grows as you play quieter.
STORM_GUST_DEPTH = 1.5      # loud -> up to +150% gust depth (bigger gusts)
STORM_GUST_TAU = 0.5        # loud -> gusts up to 50% faster
STORM_HIGH = 2.0            # loud -> up to +200% high-band gain (top-end howl)
STORM_DRIVE = 1.5           # loud -> up to +150% drive (grit)
STORM_MIDQ = 1.0            # loud -> up to +100% on the mid-Q ramp (whistlier)
STORM_SOFT_DARK = 0.6       # very soft -> mid + high bands down up to 60% (muffled)
STORM_SOFT_BODY = 0.25      # very soft -> up to +25% low-band body (rounded)
# gust_bright: on top of the level, a gust raises the mid band by gust^1.5 and the
# high band by gust^3 (at gust_bright 1), so a +30% gust is ~+3.4 dB / +6.8 dB.
GUST_BRIGHT_MID = 1.5
GUST_BRIGHT_HIGH = 3.0
STORM_BUFFET = 1.0          # loud -> up to +100% buffet depth
GUST_SWEEP_MAX_S = 2.0      # longest gust_sweep_s
GUST_SWEEP_DIR_TAU = 10.0   # correlation time of the wind direction (s)
# Edge whistles. Each one gets, at birth: a base pitch, a threshold wind speed
# (level x gust, the same 0..1 scale as the level) and the speed range over
# which it fades in.
WHISTLE_MAX = 8
WHISTLE_F_LO, WHISTLE_F_HI = 500.0, 2800.0
WHISTLE_THRESHOLD = (0.3, 0.75)
WHISTLE_WIDTH = (0.1, 0.3)
WHISTLE_FLICKER = 0.8       # depth of the on/off flicker (the vortices lock and unlock)
WHISTLE_FLICKER_TAU = 0.15  # (s)
WHISTLE_WOBBLE = 0.01       # +-1% slow pitch wobble
WHISTLE_WOBBLE_TAU = 0.4    # (s)
WHISTLE_LOUD_MAX = 2.0      # cap of the extra loudness at high speed
WHISTLE_BOOST = 1.5         # a narrow bandpass on pink noise has low RMS (~1/sqrt(Q))

# Pipe organ "principal chorus": harmonic multiples of the root and their raw
# levels (8' 4' 2-2/3' 2' 1-3/5' 1-1/3' 1'). The 7th harmonic is off by default
# (no classical organ rank sits at 7x); organ_harm7 brings it in at up to
# ORGAN_AMP7, for just-intonation music built on it. Levels are normalized to
# sum 1 at synth time so the worst-case aligned peak stays <= 1 before any gain.
ORGAN_MULTS = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
ORGAN_AMPS = np.array([1.00, 0.55, 0.30, 0.22, 0.13, 0.10, 0.0, 0.07], dtype=np.float64)
ORGAN_AMP7 = 0.12
ORGAN_GAIN = 0.85   # headroom so the bright partials aren't crushed by the tanh
BRIGHT_TAPER = 0.7  # how far organ_brightness can pull down the partials above the root
AIR_Q = 9.0         # resonance of the "air in the tube" bandpasses (higher = more whistly)
AIR_BOOST = 4.0     # high-Q bandpass on noise has low RMS; boost it to sit with the tone
ORGAN_WAVER = 0.006 # depth of the slow random pitch waver (natural wind detune)
ORGAN_WAVER_TAU = 1.2  # pitch waver correlation time (s)
# A gust above the wind's mean also brightens and sharpens the organ (like rising
# air pressure overblowing a flue pipe), so it tracks the wind.
ORGAN_WIND_BRIGHT = 0.6
ORGAN_WIND_PITCH = 0.01
# When the organ follows the pitch, the full pitch range is squeezed onto a deep
# root spanning only ORGAN_RANGE_SEMIS semitones.
ORGAN_BASE_HZ = 220.0     # lowest organ note at octave 0
ORGAN_RANGE_SEMIS = 7.0
# Ranks: each pipe gets its own random part of the detune (so pipes don't beat in
# step), and the detune drifts slowly (so the beat speeds up and slows down).
RANK_JITTER = 0.4          # +-40% per-pipe change of organ_beat_hz
RANK_DRIFT = 0.5           # +-50% slow drift of the detune
RANK_DRIFT_TAU = 4.0       # correlation time of that drift (s)
GOLDEN = 0.6180339887      # spreads pipe pan positions evenly, in order of birth
# organ_organic: from a clean synth organ (0) to wind finding a pipe by accident
# (1). Every value below is scaled by organ_organic, so 0 is the clean organ.
# The "pressure" is the organ swell x the gust (about 0..1.3): it decides how
# each pipe speaks. Below SPEAK_LO the pipe only breathes: the fundamental is
# gone, the octave and the twelfth flicker. It speaks fully at SPEAK_LO +
# SPEAK_W. Above OVERBLOW_AT it overblows: the fundamental drops, the octave
# takes over.
SPEAK_LO, SPEAK_W = 0.15, 0.35
OVERBLOW_AT, OVERBLOW_W = 1.0, 0.3
HESITATE_OCT = 1.2         # extra octave (x2) while the pipe hesitates
HESITATE_TWELFTH = 0.8     # extra twelfth (x3) while the pipe hesitates
OVERBLOW_DROP = 0.85       # fundamental lost when the pipe overblows
OVERBLOW_OCT = 2.0         # extra octave when the pipe overblows
ORGANIC_PIPE_SPREAD = 0.15 # each pipe speaks at its own pressure, +- this
ORGANIC_FLICKER = 0.35     # each partial's own slow random level, +-35%
ORGANIC_FLICKER_TAU = 0.4  # (s)
AIR_HESITATE = 2.0         # more breath while the pipes hesitate
ORGANIC_JITTER = 0.003     # fast pitch jitter, +-5 cents (x3 while unstable)
ORGANIC_JITTER_TAU = 0.1   # (s)
CHIFF_TAU = 0.04           # decay of the chiff, the puff when a pipe starts to speak (s)
CHIFF_HARM = 3.0           # it sits near this harmonic
CHIFF_Q = 5.0
CHIFF_GAIN = 20.0          # a bandpass on pink noise has a low RMS
CHIFF_HYST = 0.1           # the pressure must go this far over SPEAK_LO to fire
CHIFF_MIN = 1e-3           # below this the chiff is over
TUBE_LEVEL = 1.0           # blown tube level next to the sine pipe
TUBE_F_MAX = 2000.0        # pipes above this stay sines (Hz)
# Wind chord: the wind noise rings at the chord notes. A narrow bandpass on noise
# has a low RMS, so it gets a boost that grows with Q (bandwidth ~ f / Q).
WIND_CHORD_BOOST = 1.6
WIND_CHORD_GUST = 1.0      # extra gust swell on the chord: it comes and goes with the gusts

# Speaker protection on the final bus.
SUBSONIC_HP_ORDER = 4      # 24 dB/oct: reduces, not just flattens, excursion below cutoff
OUTPUT_CEILING = 0.89      # limiter target peak (~ -1 dBFS)
OUTPUT_HARD_CEIL = 0.99    # absolute hard-clip backstop (last resort)
LIMITER_RELEASE_S = 0.1    # how fast the limiter gain recovers after a peak passes


def pitch_to_freq(x: float) -> float:
    """Map a 0..1 pitch position onto FREQ_LO..FREQ_HI (log)."""
    x = max(0.0, min(1.0, x))
    return FREQ_LO * (FREQ_HI / FREQ_LO) ** x


def _smooth01(x: float) -> float:
    """Smoothstep of x clipped to 0..1: 0 below 0, 1 above 1, a soft S between."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def build_biquad_bandpass(fc: float, Q: float, sr: int):
    """RBJ bandpass (constant skirt, peak gain = Q)."""
    w0 = 2.0 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * Q)
    b = np.array([alpha, 0.0, -alpha], dtype=np.float64)
    a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha], dtype=np.float64)
    return b / a[0], a / a[0]


class WindOrgan:
    """One synth voice with its own filter, LFO and reverb state.

    Not thread-safe: call it from the audio thread only. Read-only outputs for a
    UI: `tilt` (0..1 pitch position of the last block) and `organ_swell` (0..1)."""

    def __init__(self, sr: int = SR, seed: int | None = None):
        self.sr = sr
        self.rng = np.random.default_rng(seed)
        self.reverb = StereoReverb(sr)

        # All filter states carry two columns (left/right noise), filtered on axis 0.
        self.pink_zis = [np.zeros((1, 2)) for _ in PINK_POLES]
        self.rumble_b, self.rumble_a = iirfilter(
            2, RUMBLE_CUTOFF / (sr / 2.0), btype="low", ftype="butter")
        self.rumble_zi = np.zeros((2, 2))
        self.bp_zi = np.zeros((2, 2))
        self.low_zi = np.zeros((2, 2))
        self.mid_zi = np.zeros((2, 2))
        self.high_zi = np.zeros((2, 2))
        self.gust_state = 0.0
        self.amp_prev = None  # wind level at the end of the last block
        # The new wind layers (buffet, ...) draw from their own generator, so
        # the random stream of the older layers stays the same when they are off.
        self.wind_rng = np.random.default_rng(None if seed is None else seed + 104729)
        self.buffet_hz = 0.0  # cutoff of the filter below (0 = not built yet)
        self.buffet_b = self.buffet_a = self.buffet_norm = None
        self.buffet_zi = np.zeros((2, 2))
        self.gust_hist = None  # gust_sweep: the gust of the last blocks (a ring)
        self.gust_hist_pos = 0
        self.gust_dir_state = 0.0
        self.whistles: list = []  # _Whistle, born when Params.whistles goes up
        self.whistle_rng = np.random.default_rng(None if seed is None else seed + 15485863)
        self.gust_extra_states = [0.0, 0.0, 0.0]  # fractal layers (tau/2, /4, /8)
        self.q_drift_state = 0.0

        # zi persists across blocks so a pitch move doesn't click.
        self.bourd_root_zi = np.zeros((2, 2))
        self.bourd_fifth_zi = np.zeros((2, 2))
        self.bourd_third_zi = np.zeros((2, 2))

        # One _Pipe per chord ratio, kept while it fades out. Their random draws
        # use their own generator, so the wind's random stream (and a seeded
        # render without a chord) stays the same as before chords existed.
        self.pipes: dict = {}
        self.pipe_count = 0
        self.pipe_rng = np.random.default_rng(None if seed is None else seed + 7919)
        self.chord_text = ""
        self.chord = parse_chord("")
        self.rings: dict = {}  # wind chord: ratio -> [filter state, fade gain]
        self.trem_phase = 0.0
        self.organ_waver_state = 0.0
        self.organ_swell = 0.0  # slow envelope built by sustained strong wind
        self.organ_amp_prev = None  # organ level at the end of the last block
        self.organic_rng = np.random.default_rng(None if seed is None else seed + 32452843)
        self.organic_prev = None
        self.tube_tone_hz = 0.0   # organ_tone_hz lowpass on the tubes, built on use
        self.tube_tone_ba = None
        self.tube_tone_zi = np.zeros((2, 2))
        self.organ_air_zi = [np.zeros((2, 2)) for _ in range(3)]  # air bands (f0, 2f0, 3f0)

        # The subsonic cutoff is a knob, so its SOS is rebuilt only when it changes;
        # the section count (hence the zi shape) is fixed by the order.
        self.sub_hp_hz = 30.0
        self.sub_sos = self._build_sub_hp(self.sub_hp_hz)
        self.sub_zi = np.zeros((self.sub_sos.shape[0], 2, 2))
        self.limiter_gain = 1.0

        self.tilt = 0.0

    def _build_sub_hp(self, fc: float):
        return butter(SUBSONIC_HP_ORDER, max(5.0, min(fc, self.sr * 0.45)) / (self.sr / 2.0),
                      btype="high", output="sos")

    def _lfo_step(self, prev: float, alpha: float, depth: float,
                  rng: np.random.Generator | None = None) -> tuple[float, float]:
        """One block of a unit-variance random-walk LFO. Returns (new_state, mod)
        where mod is in [1-depth, 1+depth]. The norm rescales because a one-pole on
        unit-variance noise has stationary stddev sqrt(alpha/(2-alpha)). rng: the
        generator to draw from (default: the wind's own)."""
        rng = self.rng if rng is None else rng
        new_state = (1.0 - alpha) * prev + alpha * rng.standard_normal()
        norm = math.sqrt((2.0 - alpha) / alpha)
        return new_state, 1.0 + depth * math.tanh(new_state * norm * 0.7)

    def render(self, frames: int, freq: float, level: float, p: Params) -> np.ndarray:
        """The full chain for one block. Returns (frames, 2) float64 in [-1, 1]."""
        out = self.room(self.voice(frames, freq, level, p), p)
        if p.wind_gain != 1.0:
            np.clip(out, -1.0, 1.0, out)
        return self.protect(out, p)

    def voice(self, frames: int, freq: float, level: float, p: Params) -> np.ndarray:
        """Wind + bourdon + organ, saturated. Returns (frames, 2)."""
        sr = self.sr
        rng = self.rng
        f = max(60.0, min(sr * 0.45, freq))
        amp = max(0.0, min(1.0, level))

        # Storm morph: the level drives the wind's CHARACTER, not just its loudness.
        # `m` is biased to the loud end (amp^1.5) so soft play stays calm; `soft`
        # grows as you play quieter. At storm=0 every value below is the plain one.
        s = p.storm
        m = s * (amp ** 1.5)
        soft = s * (1.0 - amp)
        drive_eff = p.drive * (1.0 + STORM_DRIVE * m)

        # Two independent white noises for left/right, correlated by rho. A linear
        # mix of independent noises stays flat at every width, in stereo AND summed
        # to mono (mono only loses up to 3 dB, never a notch). Both columns run the
        # SAME filters below: same timbre, independent grain.
        rho = 1.0 - max(0.0, min(1.0, p.stereo_width))
        nL = rng.standard_normal(frames).astype(np.float64)
        nR = rho * nL + math.sqrt(max(0.0, 1.0 - rho * rho)) * rng.standard_normal(frames)
        white = np.stack([nL, nR], axis=1) * 0.4   # (frames, 2)
        src = white * PINK_DIRECT
        for i, (pole, gain) in enumerate(zip(PINK_POLES, PINK_GAINS)):
            y, self.pink_zis[i] = lfilter([gain], [1.0, -pole], white, axis=0, zi=self.pink_zis[i])
            src = src + y
        src *= PINK_SCALE

        # The gust LFO is the wind's slow breath. Compute it when the wind OR the
        # organ needs it, so the organ can follow the same gusts.
        if p.use_gust or p.organ_mode:
            # storm: bigger and faster gusts when loud, gentle slow breath when soft.
            gust_depth_eff = min(0.95, p.gust_depth * (1.0 + STORM_GUST_DEPTH * m))
            gust_tau_eff = p.gust_tau_s * (1.0 - STORM_GUST_TAU * m)
            gust_alpha = 1.0 - math.exp(-(frames / sr) / max(gust_tau_eff, 0.05))
            self.gust_state, gust_mod = self._lfo_step(self.gust_state, gust_alpha, gust_depth_eff)
            # Fractal turbulence: faster, shallower layers an octave apart with
            # 1/f-decaying depth, summed as deviations around 1.
            for li in range(min(int(p.gust_layers) - 1, len(self.gust_extra_states))):
                scale = 2.0 ** (li + 1)
                sub_alpha = 1.0 - math.exp(-(frames / sr) / max(gust_tau_eff / scale, 0.02))
                self.gust_extra_states[li], sub_mod = self._lfo_step(
                    self.gust_extra_states[li], sub_alpha, gust_depth_eff / scale)
                gust_mod += sub_mod - 1.0
            gust_mod = 1.0 + max(-0.95, min(0.95, gust_mod - 1.0))
        else:
            gust_mod = 1.0
        # The level and the gust change once per block. Ramp the wind level
        # across the block, so it glides instead of stepping ~94 times a second.
        if p.use_gust and p.gust_sweep_s > 0.0:
            amp_target = amp * self._gust_sweep(frames, gust_mod, p)    # (2,): left, right
        else:
            amp_target = amp * gust_mod if p.use_gust else amp
        amp_prev = amp_target if self.amp_prev is None else self.amp_prev
        self.amp_prev = amp_target
        amp_eff = np.linspace(amp_prev, amp_target, frames)   # (frames,) or (frames, 2)
        if amp_eff.ndim == 1:
            amp_eff = amp_eff[:, None]
        if p.buffet_depth > 0.0:
            amp_eff = amp_eff * self._buffet(frames, m, p)               # (frames, 2)

        # gust_bright: faster air makes more high-frequency noise, so the mid and
        # high bands follow a gust more than the low rumble does.
        bright_up = 1.0
        if p.gust_bright > 0.0 and p.use_gust:
            bright_up = max(0.05, gust_mod) ** p.gust_bright

        if p.q_drift_depth > 0.0:
            q_alpha = 1.0 - math.exp(-(frames / sr) / max(p.q_drift_tau_s, 0.05))
            self.q_drift_state, q_mod = self._lfo_step(self.q_drift_state, q_alpha, p.q_drift_depth)
        else:
            q_mod = 1.0

        # Each layer goes into `comp` so Params.solo can isolate any subset.
        comp: dict[str, np.ndarray] = {}

        tilt = (math.log(f) - math.log(FREQ_LO)) / (math.log(FREQ_HI) - math.log(FREQ_LO))
        tilt = max(0.0, min(1.0, tilt))
        self.tilt = tilt

        if p.use_3band:
            mid_fc = p.mid_fc_lo * (p.mid_fc_hi / p.mid_fc_lo) ** tilt
            mid_Q = (1.2 + p.mid_q_max * (amp ** 0.6) * (1.0 + STORM_MIDQ * m)) * q_mod

            # Eolian coupling: a gust lifts the formant centers (mid + high only, not
            # the played pitch), so the whistle rises as it swells.
            high_fc_eff = p.high_fc
            if p.eolian_depth > 0.0:
                lift = 1.0 + p.eolian_depth * (gust_mod - 1.0)
                mid_fc = max(20.0, min(sr * 0.45, mid_fc * lift))
                high_fc_eff = max(20.0, min(sr * 0.45, p.high_fc * lift))

            low_b, low_a = build_biquad_bandpass(p.low_fc, p.low_q, sr)
            mid_b, mid_a = build_biquad_bandpass(mid_fc, mid_Q, sr)
            high_b, high_a = build_biquad_bandpass(high_fc_eff, p.high_q, sr)

            low, self.low_zi = lfilter(low_b, low_a, src, axis=0, zi=self.low_zi)
            mid, self.mid_zi = lfilter(mid_b, mid_a, src, axis=0, zi=self.mid_zi)
            high, self.high_zi = lfilter(high_b, high_a, src, axis=0, zi=self.high_zi)

            # Muffle the mid AND high bands for the soft breeze. The high band is
            # pitch-gated (tilt**3 ~ 0 at low pitch), so the mid band, which is the
            # body you hear there, has to be darkened too.
            g_dark = 1.0 - STORM_SOFT_DARK * soft
            g_low = ((1.0 - tilt) * 0.6 + 0.3) * (1.0 + STORM_SOFT_BODY * soft)
            g_mid = 0.7 * g_dark
            g_high = (tilt ** 3) * p.high_band_gain * (1.0 + STORM_HIGH * m) * g_dark
            if bright_up != 1.0:
                g_mid *= bright_up ** GUST_BRIGHT_MID
                g_high *= bright_up ** GUST_BRIGHT_HIGH

            comp["low"] = low * g_low * 4.0 * amp_eff
            comp["mid"] = mid * g_mid * 1.5 * amp_eff
            comp["high"] = high * g_high * 1.5 * amp_eff
        else:
            Q = (1.2 + (p.mid_q_max + 1.0) * (amp ** 0.6) * (1.0 + STORM_MIDQ * m)) * q_mod
            b, a = build_biquad_bandpass(f, Q, sr)
            bp, self.bp_zi = lfilter(b, a, src, axis=0, zi=self.bp_zi)
            rumble, self.rumble_zi = lfilter(self.rumble_b, self.rumble_a, src, axis=0,
                                             zi=self.rumble_zi)
            # Single-band isn't split into bands; expose it as "low" so solo still finds it.
            if bright_up != 1.0:
                bp = bp * bright_up ** GUST_BRIGHT_MID
            comp["low"] = (bp * 1.5 * 0.75 + rumble * 4.0 * 0.55) * amp_eff

        # Bourdon: narrow bandpass on the same noise at root + optional 5th/3rd.
        # A pitched whistle (like air across a bottle), not a sine pad.
        if p.tone_level > 0.0:
            def bvoice(fv: float, zi):
                b, a = build_biquad_bandpass(fv, p.bourdon_q, sr)
                return lfilter(b, a, src, axis=0, zi=zi)

            voices, self.bourd_root_zi = bvoice(f, self.bourd_root_zi)
            n_voices = 1
            if p.use_fifth and f * 1.5 < sr * 0.45:
                fifth, self.bourd_fifth_zi = bvoice(f * 1.5, self.bourd_fifth_zi)
                voices = voices + fifth
                n_voices += 1
            if p.third_mode > 0:
                ratio = 1.2 if p.third_mode == 1 else 1.25  # 6:5 minor, 5:4 major
                if f * ratio < sr * 0.45:
                    third, self.bourd_third_zi = bvoice(f * ratio, self.bourd_third_zi)
                    voices = voices + third
                    n_voices += 1

            # High-Q bandpass on noise has low RMS; boost so the whistle sits with the wind.
            comp["bourdon"] = (voices / n_voices) * amp_eff * p.tone_level * 4.0

        # Wind chord: the same noise rings at the chord notes, at the level of the
        # wind and a bit more with the gusts. It works without the organ.
        if p.wind_chord > 0.0:
            ring_gain = amp_eff * max(0.0, 1.0 + WIND_CHORD_GUST * (gust_mod - 1.0))
            comp["chord"] = self._wind_chord(frames, self._root(f, p), src, ring_gain, p)

        # Edge whistles: they sing only in strong wind, and higher when it blows harder.
        if p.whistles > 0 or self.whistles:
            w = self._whistles(frames, amp * gust_mod, src, p)
            if w is not None:
                comp["whistle"] = w

        if p.organ_mode:
            comp["organ"] = self._organ(frames, f, amp, gust_mod, src, p)

        keys = comp.keys() & p.solo if p.solo else comp.keys()
        mix = np.zeros((frames, 2), dtype=np.float64)
        for k in keys:
            mix = mix + comp[k]
        return np.tanh(mix * drive_eff)

    def _gust_sweep(self, frames: int, gust_mod: float, p: Params) -> np.ndarray:
        """The gust for the left and the right side, (2,). A gust reaches one side
        first: the other side gets the gust of up to gust_sweep_s ago. The wind
        direction drifts slowly, so the sweep changes side and length, and with
        the wind from the front both sides get the same gust."""
        sr = self.sr
        n = int(math.ceil(GUST_SWEEP_MAX_S * sr / frames)) + 2
        if self.gust_hist is None or len(self.gust_hist) != n:
            self.gust_hist = np.full(n, gust_mod)   # first block, or a new block size
            self.gust_hist_pos = 0
        pos = (self.gust_hist_pos + 1) % n
        self.gust_hist[pos] = gust_mod
        self.gust_hist_pos = pos
        alpha = 1.0 - math.exp(-(frames / sr) / GUST_SWEEP_DIR_TAU)
        self.gust_dir_state, d = self._lfo_step(self.gust_dir_state, alpha, 1.0, self.wind_rng)
        direction = d - 1.0   # -1..1: -1 = from the right, 0 = from the front, 1 = from the left
        delay = min(GUST_SWEEP_MAX_S, p.gust_sweep_s) * abs(direction) * sr / frames  # blocks
        k = int(delay)
        a = self.gust_hist[(pos - k) % n]
        b = self.gust_hist[(pos - k - 1) % n]
        late = a + (b - a) * (delay - k)
        return np.array([gust_mod, late]) if direction >= 0.0 else np.array([late, gust_mod])

    def _whistles(self, frames: int, speed: float, src: np.ndarray,
                  p: Params) -> np.ndarray | None:
        """Aeolian tones: the wind on a wire, a crack, an edge. Each whistle has
        its own pitch, its own place and its own threshold speed. Above it the
        whistle comes in, rises in pitch with the speed (Strouhal: f ~ speed) and
        flickers (the vortices lock on and off). Returns (frames, 2), or None
        when all whistles are silent."""
        sr = self.sr
        count = max(0, min(WHISTLE_MAX, int(p.whistles)))
        while len(self.whistles) < count:
            self.whistles.append(_Whistle(len(self.whistles), self.whistle_rng))
        q = max(2.0, p.whistle_q)
        tau_alpha = 1.0 - math.exp(-(frames / sr) / WHISTLE_FLICKER_TAU)
        wob_alpha = 1.0 - math.exp(-(frames / sr) / WHISTLE_WOBBLE_TAU)
        out = np.zeros((frames, 2), dtype=np.float64)
        alive = False
        for i, w in enumerate(self.whistles):
            over = speed / w.threshold
            on = (speed - w.threshold) / w.width
            on = 0.0 if i >= count else max(0.0, min(1.0, on))
            on = on * on * (3.0 - 2.0 * on)
            w.flicker, fl = self._lfo_step(w.flicker, tau_alpha, WHISTLE_FLICKER, self.whistle_rng)
            w.wobble, wb = self._lfo_step(w.wobble, wob_alpha, WHISTLE_WOBBLE, self.whistle_rng)
            # Aeolian sound grows fast with the speed: louder as it blows harder.
            g = on * max(0.0, fl) * min(WHISTLE_LOUD_MAX, over ** 1.5)
            g0, w.gain = w.gain, g
            if g0 <= 0.0 and g <= 0.0:
                continue
            alive = True
            fc = w.f_base * max(1.0, over) ** max(0.0, p.whistle_track) * wb
            fc = max(20.0, min(sr * 0.45, fc))
            b, a = build_biquad_bandpass(fc, q, sr)
            y, w.zi = lfilter(b, a, src[:, 0], zi=w.zi)
            y = y * np.linspace(g0, g, frames)
            theta = (w.pos + 1.0) * (math.pi / 4.0)
            out[:, 0] += y * (math.sqrt(2.0) * math.cos(theta))
            out[:, 1] += y * (math.sqrt(2.0) * math.sin(theta))
        if not alive:
            return None
        return out * (WHISTLE_BOOST * math.sqrt(q) * p.whistle_level)

    def _buffet(self, frames: int, m: float, p: Params) -> np.ndarray:
        """The fast flutter of real wind on the level: noise lowpassed at
        buffet_hz, one value per sample, around 1. Left and right are correlated
        like the wind noise (stereo_width). Deeper in a storm. Returns (frames, 2)."""
        sr = self.sr
        fc = max(0.5, min(200.0, p.buffet_hz))
        if fc != self.buffet_hz:
            self.buffet_b, self.buffet_a = butter(2, fc / (sr / 2.0))
            # Scale to unit variance on white noise: divide by the RMS of the
            # impulse response (its tail is gone after ~20 periods of fc).
            imp = np.zeros(int(20.0 * sr / fc))
            imp[0] = 1.0
            h = lfilter(self.buffet_b, self.buffet_a, imp)
            self.buffet_norm = 1.0 / math.sqrt(float(np.sum(h * h)))
            self.buffet_hz = fc
        rho = 1.0 - max(0.0, min(1.0, p.stereo_width))
        w = self.wind_rng.standard_normal((frames, 2))
        w[:, 1] = rho * w[:, 0] + math.sqrt(max(0.0, 1.0 - rho * rho)) * w[:, 1]
        y, self.buffet_zi = lfilter(self.buffet_b, self.buffet_a, w, axis=0, zi=self.buffet_zi)
        depth = min(0.9, p.buffet_depth * (1.0 + STORM_BUFFET * m))
        return 1.0 + depth * np.tanh(y * (self.buffet_norm * 0.7))

    def _organ(self, frames: int, f: float, amp: float, gust_mod: float,
               src: np.ndarray, p: Params) -> np.ndarray:
        """Pipe-organ drone driven by the same wind you hear as noise."""
        sr = self.sr
        # gust_dev = how hard the wind blows right now vs its mean. organ_wind sets
        # how tightly the organ swells, brightens and sharpens with it.
        gust_dev = gust_mod - 1.0
        waver_alpha = 1.0 - math.exp(-(frames / sr) / ORGAN_WAVER_TAU)
        self.organ_waver_state, waver_mod = self._lfo_step(
            self.organ_waver_state, waver_alpha, ORGAN_WAVER)
        self.trem_phase = math.fmod(self.trem_phase + 2.0 * math.pi * p.trem_rate * frames / sr,
                                    2.0 * math.pi)
        trem = math.sin(self.trem_phase)

        # Swell with inertia: the organ only builds while the level sits above
        # organ_threshold, rising over organ_rise_s and fading over organ_fall_s.
        # organ_follow < 1 blends towards a steady drone that ignores the level.
        follow = max(0.0, min(1.0, p.organ_follow))
        demand = max(0.0, (amp - p.organ_threshold) / max(1e-6, 1.0 - p.organ_threshold))
        demand = (follow * demand + (1.0 - follow)) * max(0.0, min(1.0, p.organ_gate))
        swell_tau = p.organ_rise_s if demand > self.organ_swell else p.organ_fall_s
        swell_alpha = 1.0 - math.exp(-(frames / sr) / max(swell_tau, 0.05))
        self.organ_swell += (demand - self.organ_swell) * swell_alpha

        organ_amp_eff = ((follow * amp + 1.0 - follow) * p.organ_level * self.organ_swell
                         * (1.0 + p.organ_wind * gust_dev) * (1.0 + p.trem_depth * trem))
        bright = min(1.0, max(0.0, p.organ_brightness + ORGAN_WIND_BRIGHT * p.organ_wind * gust_dev))

        # The waver, the vibrato and the gust overblow detune the root a little, so
        # it never sits dead-flat.
        f0 = self._root(f, p) * waver_mod * (1.0 + p.trem_pitch * trem) * (1.0 + ORGAN_WIND_PITCH * p.organ_wind * gust_dev)
        f0 = max(20.0, min(sr * 0.45, f0))

        amps = ORGAN_AMPS.copy()
        amps[6] = ORGAN_AMP7 * max(0.0, min(1.0, p.organ_harm7))
        # Brightness pulls down the partials above the root; renormalize so the
        # overall level holds as it moves.
        amps[ORGAN_MULTS >= 2.0] *= (1.0 - BRIGHT_TAPER) + BRIGHT_TAPER * bright
        amps /= max(amps.sum(), 1e-9)
        # How hard the air pushes into the pipes right now (see SPEAK_LO).
        organic = max(0.0, min(1.0, p.organ_organic))
        pressure = self.organ_swell * (1.0 + p.organ_wind * gust_dev)
        tone = self._pipes(frames, f0, amps, p, pressure, organic, src)

        # "Air in the metal tube": noise resonating at the first harmonics, so the
        # breath is pitched (whistling through the pipe), not broadband hiss.
        # It rings at the lowest pipe of the chord (the root without a chord).
        air = 0.0
        if p.organ_air > 0.0:
            f_air = f0 * float(self.chord[0])
            for j, (k, w) in enumerate(((1.0, 1.0), (2.0, 0.6), (3.0, 0.4))):
                fk = f_air * k
                if fk < sr * 0.45:
                    ab, aa = build_biquad_bandpass(fk, AIR_Q, sr)
                    band, self.organ_air_zi[j] = lfilter(ab, aa, src, axis=0, zi=self.organ_air_zi[j])
                    air = air + band * w

        # The level moves once per block (swell, gust, tremulant): ramp it across
        # the block, so the 5.5 Hz tremulant is a curve, not ~17 steps.
        amp_prev = organ_amp_eff if self.organ_amp_prev is None else self.organ_amp_prev
        self.organ_amp_prev = organ_amp_eff
        ramp = np.linspace(amp_prev, organ_amp_eff, frames)[:, None]
        # The tone is centred unless organ_spread; the breath is the stereo noise.
        if organic > 0.0 and p.organ_air > 0.0:
            air = air * (1.0 + organic * AIR_HESITATE * (1.0 - _smooth01((pressure - SPEAK_LO) / SPEAK_W)))
        return (tone * ORGAN_GAIN + air * AIR_BOOST * p.organ_air) * ramp

    @staticmethod
    def _root(f: float, p: Params) -> float:
        """Root of the organ and of the wind chord: one fixed note in hold mode,
        else the pitch squeezed onto ORGAN_RANGE_SEMIS semitones above a deep base."""
        if p.organ_hold:
            return 440.0 * 2.0 ** ((p.organ_note - 69.0) / 12.0)
        ratio = (math.log(f) - math.log(FREQ_LO)) / (math.log(FREQ_HI) - math.log(FREQ_LO))
        ratio = max(0.0, min(1.0, ratio))
        return ORGAN_BASE_HZ * (2.0 ** p.organ_octave) * (2.0 ** (ratio * ORGAN_RANGE_SEMIS / 12.0))

    def _update_chord(self, p: Params) -> None:
        if p.organ_chord != self.chord_text:
            try:
                self.chord = parse_chord(p.organ_chord)
            except (ValueError, ZeroDivisionError):
                pass  # bad text: keep the last good chord
            self.chord_text = p.organ_chord

    def _wind_chord(self, frames: int, root: float, src: np.ndarray, gain: float,
                    p: Params) -> np.ndarray:
        """The wind noise through one narrow bandpass per chord note. Noise, not a
        tone: it sounds like the wind found the chord in the building. Each note
        fades in and out over organ_fade_s. Returns (frames, 2)."""
        sr = self.sr
        self._update_chord(p)
        for r in self.chord:
            if r not in self.rings:
                self.rings[r] = [np.zeros((2, 2)), 0.0]
        fade = frames / (sr * max(p.organ_fade_s, 0.01))
        q = max(1.0, p.wind_chord_q)
        boost = WIND_CHORD_BOOST * math.sqrt(q / len(self.chord))
        out = np.zeros((frames, 2), dtype=np.float64)
        for r, ring in list(self.rings.items()):
            g0 = ring[1]
            g1 = g0 + max(-fade, min(fade, (1.0 if r in self.chord else 0.0) - g0))
            ring[1] = g1
            if g0 <= 0.0 and g1 <= 0.0:
                del self.rings[r]
                continue
            fr = root * float(r)
            if not 20.0 <= fr < sr * 0.45:
                continue
            b, a = build_biquad_bandpass(fr, q, sr)
            y, ring[0] = lfilter(b, a, src, axis=0, zi=ring[0])
            g = np.linspace(g0, g1, frames)
            out += y * (g * g * (3.0 - 2.0 * g))[:, None]
        return out * (boost * gain * p.wind_chord)

    def _pipe_speech(self, pipe: "_Pipe", pressure: float, organic: float,
                     frames: int) -> np.ndarray:
        """Weights for the partials of one pipe (all ORGAN_MULTS): how it speaks
        at this pressure, and the slow flicker of each partial. Real pipes in
        moving air are never as steady as sines."""
        if pipe.p_off is None:
            pipe.p_off = self.organic_rng.uniform(-1.0, 1.0) * ORGANIC_PIPE_SPREAD
        pp = pressure + pipe.p_off * organic
        speak = _smooth01((pp - SPEAK_LO) / SPEAK_W)
        # Chiff: when the pressure crosses the speaking point upwards, the pipe
        # starts with a puff of noise. It fires again only after the pressure
        # went back under SPEAK_LO.
        if pipe.chiff_armed is None:
            pipe.chiff_armed = pp < SPEAK_LO + CHIFF_HYST
        if pipe.chiff_armed and pp > SPEAK_LO + CHIFF_HYST:
            pipe.chiff, pipe.chiff_armed = 1.0, False
        elif pp < SPEAK_LO:
            pipe.chiff_armed = True
        over = _smooth01((pp - OVERBLOW_AT) / OVERBLOW_W)
        pipe.pressure, pipe.over = pp, over   # the tube of this pipe blows with them
        w = np.ones(len(ORGAN_MULTS))
        w[0] = max(0.0, 1.0 - organic * (1.0 - speak) - organic * OVERBLOW_DROP * over)
        w[1] = 1.0 + organic * (HESITATE_OCT * (1.0 - speak) + OVERBLOW_OCT * over)
        w[2] = 1.0 + organic * HESITATE_TWELFTH * (1.0 - speak)
        # The flicker is deeper while the pipe is unstable (hesitating, overblowing).
        depth = min(0.9, organic * ORGANIC_FLICKER * (1.0 + (1.0 - speak) + over))
        alpha = 1.0 - math.exp(-(frames / self.sr) / ORGANIC_FLICKER_TAU)
        pipe.flicker = (1.0 - alpha) * pipe.flicker + alpha * self.organic_rng.standard_normal(len(w))
        norm = math.sqrt((2.0 - alpha) / alpha)
        # Fast pitch jitter: the turbulence in the jet, a shimmer of a few cents.
        jit_alpha = 1.0 - math.exp(-(frames / self.sr) / ORGANIC_JITTER_TAU)
        pipe.jitter_state, pipe.pitch_mod = self._lfo_step(
            pipe.jitter_state, jit_alpha, ORGANIC_JITTER * organic * (1.0 + (1.0 - speak) + over),
            self.organic_rng)
        return w * (1.0 + depth * np.tanh(pipe.flicker * (norm * 0.7)))

    def _pipes(self, frames: int, f0: float, amps: np.ndarray, p: Params,
               pressure: float = 1.0, organic: float = 0.0,
               src: np.ndarray | None = None) -> np.ndarray:
        """All the pipes of the chord, each as organ_ranks detuned copies, panned.
        A pipe fades in or out over organ_fade_s when the chord changes. Returns
        (frames, 2). One pipe, one rank, no spread = the classic mono drone."""
        sr = self.sr
        self._update_chord(p)
        ranks = max(1, int(p.organ_ranks))
        first = not self.pipes  # the very first pipe starts at full gain, like before
        for r in self.chord:
            if r not in self.pipes:
                self.pipes[r] = _Pipe(self.pipe_count, self.pipe_rng, first)
                self.pipe_count += 1

        fade = frames / (sr * max(p.organ_fade_s, 0.01))
        norm = 1.0 / math.sqrt(len(self.chord) * ranks)
        spread = max(0.0, min(1.0, p.organ_spread))
        drift_alpha = 1.0 - math.exp(-(frames / sr) / RANK_DRIFT_TAU)
        n = np.arange(frames, dtype=np.float64)
        glide = n * (n - 1.0) / (2.0 * frames)   # sum of the ramp n/frames up to n
        # organ_organic crossfades the sine pipes into blown tubes, with equal
        # power (they are not phase-locked). Ramped across the block.
        o0 = organic if self.organic_prev is None else self.organic_prev
        self.organic_prev = organic
        w_sine = np.cos(np.linspace(o0, organic, frames) * (math.pi / 2.0))
        w_tube = np.sin(np.linspace(o0, organic, frames) * (math.pi / 2.0))
        sines = max(o0, organic) < 1.0
        tubes = max(o0, organic) > 0.0 and src is not None
        out = np.zeros((frames, 2), dtype=np.float64)
        tube_out = np.zeros((frames, 2), dtype=np.float64) if tubes else None
        for r, pipe in list(self.pipes.items()):
            target = 1.0 if r in self.chord else 0.0
            g0 = pipe.gain
            g1 = g0 + max(-fade, min(fade, target - g0))
            pipe.gain = g1
            if g0 <= 0.0 and g1 <= 0.0:
                del self.pipes[r]
                continue
            f = f0 * float(r)
            if f >= sr * 0.45:
                continue
            pipe.set_ranks(ranks)
            if ranks > 1:
                pipe.drift, drift_mod = self._lfo_step(pipe.drift, drift_alpha, RANK_DRIFT)
                offs = p.organ_beat_hz * pipe.jitter * drift_mod * np.linspace(-0.5, 0.5, ranks)
            else:
                offs = np.zeros(1)
            keep = ((ORGAN_MULTS * f) < (sr * 0.45)) & (amps > 0.0)
            keep[0] = True  # the fundamental always plays
            a_now = amps
            if p.organ_tone_hz > 0.0:
                a_now = a_now / (1.0 + (ORGAN_MULTS * f / p.organ_tone_hz) ** 2)
            # The partial levels move per block (the brightness follows the gusts):
            # ramp each one across the block.
            a_prev = a_now if pipe.amps is None else pipe.amps
            if organic > 0.0:
                # The pipe's valve opens with its fade: a new chord pipe hesitates
                # before it speaks.
                a_now = a_now * self._pipe_speech(pipe, pressure * _smooth01(g1), organic, frames)
            pipe.amps = a_now
            f_now = f * pipe.pitch_mod if organic > 0.0 else f

            # Pan: the pipe sits at its own place and its ranks spread around it,
            # so the beat moves between the speakers. Equal power, centre = 1 / 1.
            rank_pos = np.linspace(-1.0, 1.0, ranks) if ranks > 1 else np.zeros(1)
            pan = spread * np.clip(0.5 * pipe.pos + 0.5 * rank_pos, -1.0, 1.0)
            theta = (pan + 1.0) * (math.pi / 4.0)
            gl, gr = math.sqrt(2.0) * np.cos(theta), math.sqrt(2.0) * np.sin(theta)
            g = np.linspace(g0, g1, frames)
            g = g * g * (3.0 - 2.0 * g) * norm   # smoothstep: soft start and end of a fade

            # A very high pipe keeps its sine: a tube up there costs a lot of CPU
            # (a very short delay) and would only add hiss.
            blown = tubes and f < TUBE_F_MAX
            if sines or not blown:
                mults = ORGAN_MULTS[keep]
                a = np.linspace(a_prev[keep], a_now[keep], frames, axis=1)   # (partials, frames)
                # One fundamental phase per rank, carried across blocks; harmonic k
                # is exactly k * phase, so the partials stay phase-locked at block
                # joins. The frequency glides from the last block's to this one's:
                # the phase is the running sum of a linear ramp of the step.
                inc = 2.0 * math.pi * (f_now + offs) / sr                 # (ranks,)
                inc0 = inc if pipe.inc is None or len(pipe.inc) != ranks else pipe.inc
                pipe.inc = inc
                dinc = (inc - inc0)[:, None]
                ph = pipe.phase[:, None] + inc0[:, None] * n[None, :] + dinc * glide[None, :]
                arg = mults[:, None, None] * ph[None]
                wave = np.einsum("kn,krn->rn", a, np.sin(arg))
                pipe.phase = np.fmod(pipe.phase + inc0 * frames + (inc - inc0) * (frames - 1) / 2.0,
                                     2.0 * math.pi)
                # Panning alone keeps L/R ~0.64 correlated. Spaced mics in a room
                # hear each pipe with a phase offset between the ears, so each rank
                # gets one too (at most +-90 degrees: mono loses 3 dB at worst).
                if spread > 0.0:
                    shift = spread * pipe.lr_shift
                    quad = np.einsum("kn,krn->rn", a, np.cos(arg))
                    wave_r = np.cos(shift)[:, None] * wave + np.sin(shift)[:, None] * quad
                else:
                    wave_r = wave
                gs = g if (organic == 0.0 and o0 == 0.0) or not blown else g * w_sine
                out[:, 0] += gl @ wave * gs
                out[:, 1] += gr @ wave_r * gs

            if blown:
                # The same pipe, blown: its tube gets the pipe's pressure (swell,
                # gust, its own fade). The pressure picks its mode (tube.py).
                if pipe.tube is None:
                    pipe.tube = Tube(sr, ranks, self.organic_rng)
                pipe.tube.set_ranks(ranks)
                tp0 = pipe.pressure if pipe.tube_pressure is None else pipe.tube_pressure
                pipe.tube_pressure = pipe.pressure
                y = pipe.tube.process(f_now + offs, tp0, pipe.pressure, src, p.organ_cavity)
                gt = g * w_tube * TUBE_LEVEL
                tube_out[:, 0] += gl @ y * gt
                tube_out[:, 1] += gr @ y * gt
            else:
                pipe.tube = None   # it starts fresh (silent) next time
                pipe.tube_pressure = None

            if pipe.chiff > CHIFF_MIN and src is not None and organic > 0.0:
                # The chiff: noise near the 3rd harmonic, a fast decay.
                decay = math.exp(-frames / (sr * CHIFF_TAU))
                env = pipe.chiff * decay ** (n / frames)
                pipe.chiff *= decay
                fc = max(20.0, min(sr * 0.45, f * CHIFF_HARM))
                cb, ca = build_biquad_bandpass(fc, CHIFF_Q, sr)
                puff, pipe.chiff_zi = lfilter(cb, ca, src[:, 0], zi=pipe.chiff_zi)
                puff = puff * env * g * (CHIFF_GAIN * organic)
                out[:, 0] += puff * float(gl.mean())
                out[:, 1] += puff * float(gr.mean())
        if tubes:
            # organ_tone_hz softens the tubes too (12 dB/oct), like the sine partials.
            if p.organ_tone_hz > 0.0:
                fc = min(p.organ_tone_hz, sr * 0.45)
                if fc != self.tube_tone_hz:
                    self.tube_tone_ba = butter(2, fc / (sr / 2.0))
                    self.tube_tone_hz = fc
                tube_out, self.tube_tone_zi = lfilter(*self.tube_tone_ba, tube_out, axis=0,
                                                      zi=self.tube_tone_zi)
            out += tube_out
        return out

    def room(self, sat: np.ndarray, p: Params) -> np.ndarray:
        """Reverb (wet/dry crossfade) then wind_gain. No clipping here."""
        if p.reverb_mix > 0.0:
            wet = self.reverb.process(sat, p.reverb_room, p.reverb_damping, p.reverb_bass)
            out = sat * (1.0 - p.reverb_mix) + wet * p.reverb_mix
        else:
            out = sat
        if p.wind_gain != 1.0:
            out = out * p.wind_gain
        return out

    def protect(self, out: np.ndarray, p: Params) -> np.ndarray:
        """Speaker protection on the final bus: strip the inaudible sub that
        over-drives the woofer, then limit the peak under the DAC ceiling so a loud
        gust never hard-clips into the tweeters. The limiter drops gain at once on a
        peak and releases slowly, ramped across the block so it never clicks."""
        frames = out.shape[0]
        if p.subsonic_hp_hz != self.sub_hp_hz:
            self.sub_sos = self._build_sub_hp(p.subsonic_hp_hz)
            self.sub_hp_hz = p.subsonic_hp_hz
        out, self.sub_zi = sosfilt(self.sub_sos, out, axis=0, zi=self.sub_zi)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        target_gain = OUTPUT_CEILING / peak if peak > OUTPUT_CEILING else 1.0
        rel_alpha = 1.0 - math.exp(-(frames / self.sr) / LIMITER_RELEASE_S)
        new_gain = (target_gain if target_gain < self.limiter_gain
                    else self.limiter_gain + (target_gain - self.limiter_gain) * rel_alpha)
        out = out * np.linspace(self.limiter_gain, new_gain, frames)[:, None]
        self.limiter_gain = new_gain
        np.clip(out, -OUTPUT_HARD_CEIL, OUTPUT_HARD_CEIL, out)
        return out


class _Whistle:
    """State of one edge whistle: where it is, when it sings, its filter."""

    def __init__(self, index: int, rng: np.random.Generator):
        self.f_base = WHISTLE_F_LO * (WHISTLE_F_HI / WHISTLE_F_LO) ** rng.uniform()
        self.threshold = rng.uniform(*WHISTLE_THRESHOLD)
        self.width = rng.uniform(*WHISTLE_WIDTH)
        self.pos = 0.8 * (((index * GOLDEN + 0.3) % 1.0) * 2.0 - 1.0)   # -0.8..0.8
        self.gain = 0.0
        self.flicker = 0.0
        self.wobble = 0.0
        self.zi = np.zeros(2)


class _Pipe:
    """State of one chord pipe: its ranks' phases, its fade gain, its place."""

    def __init__(self, index: int, rng: np.random.Generator, full: bool):
        self.gain = 1.0 if full else 0.0
        self.pos = ((index * GOLDEN) % 1.0) * 2.0 - 1.0   # -1..1, spread in birth order
        self.jitter = 1.0 + RANK_JITTER * rng.uniform(-1.0, 1.0)
        self.drift = 0.0
        self.rng = rng
        self.phase = np.zeros(1)
        self.lr_shift = rng.uniform(-0.5 * math.pi, 0.5 * math.pi, 1)
        self.amps = None   # partial levels of the last block (all ORGAN_MULTS)
        self.inc = None    # phase step per sample of each rank, last block
        # organ_organic state (see _pipe_speech). p_off is drawn on first use,
        # from another generator, so this rng's draws stay the same.
        self.p_off = None
        self.flicker = np.zeros(len(ORGAN_MULTS))
        self.jitter_state = 0.0
        self.pitch_mod = 1.0
        self.chiff = 0.0           # chiff envelope, 1 when the pipe starts to speak
        self.pressure = 0.0        # air pressure on this pipe, and how far it overblows
        self.over = 0.0
        self.tube = None           # the blown pipe (tube.py), while organ_organic > 0
        self.tube_pressure = None
        self.chiff_armed = None
        self.chiff_zi = np.zeros(2)

    def set_ranks(self, ranks: int) -> None:
        """Keep the ranks that stay; new ranks start at a random phase."""
        if len(self.phase) != ranks:
            k = min(ranks, len(self.phase))
            phase = self.rng.uniform(0.0, 2.0 * math.pi, ranks)
            shift = self.rng.uniform(-0.5 * math.pi, 0.5 * math.pi, ranks)
            phase[:k], shift[:k] = self.phase[:k], self.lr_shift[:k]
            self.phase, self.lr_shift = phase, shift
