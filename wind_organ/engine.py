"""The wind + organ synth engine. Pure DSP: no audio device, no threads, no input.

Signal flow per block:
    pink noise (Paul Kellet 6-pole IIR), two decorrelated columns (L/R) ->
        either three parallel RBJ bandpass biquads with per-band gains (the
        "spectral tilt") or one bandpass + one lowpass (single-band)
    -> multiply by level x gust LFO
    -> optional bourdon voices (narrow bandpass at root + 5th + 3rd)
    -> optional wind chord (narrow bandpasses at the organ_chord notes)
    -> optional organ drone (additive pipe partials + pitched breath)
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

    def _lfo_step(self, prev: float, alpha: float, depth: float) -> tuple[float, float]:
        """One block of a unit-variance random-walk LFO. Returns (new_state, mod)
        where mod is in [1-depth, 1+depth]. The norm rescales because a one-pole on
        unit-variance noise has stationary stddev sqrt(alpha/(2-alpha))."""
        new_state = (1.0 - alpha) * prev + alpha * self.rng.standard_normal()
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
        amp_target = amp * gust_mod if p.use_gust else amp
        amp_prev = amp_target if self.amp_prev is None else self.amp_prev
        self.amp_prev = amp_target
        amp_eff = np.linspace(amp_prev, amp_target, frames)[:, None]   # (frames, 1)

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

        if p.organ_mode:
            comp["organ"] = self._organ(frames, f, amp, gust_mod, src, p)

        keys = comp.keys() & p.solo if p.solo else comp.keys()
        mix = np.zeros((frames, 2), dtype=np.float64)
        for k in keys:
            mix = mix + comp[k]
        return np.tanh(mix * drive_eff)

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
        tone = self._pipes(frames, f0, amps, p)

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

        # The tone is centred unless organ_spread; the breath is the stereo noise.
        return (tone * ORGAN_GAIN + air * AIR_BOOST * p.organ_air) * organ_amp_eff

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

    def _pipes(self, frames: int, f0: float, amps: np.ndarray, p: Params) -> np.ndarray:
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
        out = np.zeros((frames, 2), dtype=np.float64)
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
            mults, a = ORGAN_MULTS[keep], amps[keep]
            if p.organ_tone_hz > 0.0:
                a = a / (1.0 + (mults * f / p.organ_tone_hz) ** 2)

            # One fundamental phase per rank, carried across blocks; harmonic k is
            # exactly k * phase, so the partials stay phase-locked at block joins.
            inc = 2.0 * math.pi * (f + offs) / sr                     # (ranks,)
            ph = pipe.phase[:, None] + inc[:, None] * n[None, :]      # (ranks, frames)
            arg = mults[:, None, None] * ph[None]
            wave = np.einsum("k,krn->rn", a, np.sin(arg))
            pipe.phase = np.fmod(pipe.phase + inc * frames, 2.0 * math.pi)

            # Pan: the pipe sits at its own place and its ranks spread around it,
            # so the beat moves between the speakers. Equal power, centre = 1 / 1.
            rank_pos = np.linspace(-1.0, 1.0, ranks) if ranks > 1 else np.zeros(1)
            pan = spread * np.clip(0.5 * pipe.pos + 0.5 * rank_pos, -1.0, 1.0)
            theta = (pan + 1.0) * (math.pi / 4.0)
            gl, gr = math.sqrt(2.0) * np.cos(theta), math.sqrt(2.0) * np.sin(theta)
            # Panning alone keeps L/R ~0.64 correlated. Spaced mics in a room hear
            # each pipe with a phase offset between the ears, so each rank gets
            # one too (at most +-90 degrees: mono loses 3 dB at worst, no notch).
            if spread > 0.0:
                shift = spread * pipe.lr_shift
                quad = np.einsum("k,krn->rn", a, np.cos(arg))
                wave_r = np.cos(shift)[:, None] * wave + np.sin(shift)[:, None] * quad
            else:
                wave_r = wave

            g = np.linspace(g0, g1, frames)
            g = g * g * (3.0 - 2.0 * g) * norm   # smoothstep: soft start and end of a fade
            out[:, 0] += gl @ wave * g
            out[:, 1] += gr @ wave_r * g
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

    def set_ranks(self, ranks: int) -> None:
        """Keep the ranks that stay; new ranks start at a random phase."""
        if len(self.phase) != ranks:
            k = min(ranks, len(self.phase))
            phase = self.rng.uniform(0.0, 2.0 * math.pi, ranks)
            shift = self.rng.uniform(-0.5 * math.pi, 0.5 * math.pi, ranks)
            phase[:k], shift[:k] = self.phase[:k], self.lr_shift[:k]
            self.phase, self.lr_shift = phase, shift
