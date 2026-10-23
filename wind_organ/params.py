"""All the sound knobs of the wind + organ synth, in one dataclass.

The engine reads a Params at each block. It never changes it. A host (a TUI, a
theremin, a show controller) owns its Params, changes it from any thread, and
hands the engine a copy per block (see Params.snapshot()).

The defaults are the ones of the theremin of winds, so the sound starts out the
same. load()/save() use plain JSON, and load() skips keys it doesn't know, so the
theremin's preset.json (which also holds mic and MIDI settings) loads as is.
"""
import copy
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Params:
    # --- wind voice: pink noise through bandpass filters ---
    # 3-band mode: a fixed low rumble, a mid band that follows the pitch, a high
    # whistle that comes in at high pitch. Off = one bandpass + one lowpass.
    use_3band: bool = True
    low_fc: float = 110.0          # low band center (Hz): the body of the wind
    low_q: float = 0.7
    high_fc: float = 3200.0        # high band center (Hz): the whistle on top
    high_q: float = 6.0
    high_band_gain: float = 0.45   # 0 = no top-end sizzle
    mid_fc_lo: float = 250.0       # mid band center at the lowest pitch (Hz)
    mid_fc_hi: float = 2200.0      # mid band center at the highest pitch (Hz)
    # mid_Q = 1.2 + mid_q_max * level^0.6: louder wind = narrower, more whistly mid.
    mid_q_max: float = 4.0

    # --- gusts: slow random swell on the level ---
    use_gust: bool = True
    gust_depth: float = 0.30       # +-30% level swing
    gust_tau_s: float = 1.6        # correlation time of one gust (s)
    # Extra faster, shallower gust layers an octave apart (tau/2, /4, /8), so the
    # air moves on many time scales at once. 1 = one slow swell only.
    gust_layers: int = 1
    # Strouhal coupling: a gust lifts the mid + high band centers, so the whistle
    # rises as the gust swells. 0 = fixed bands.
    eolian_depth: float = 0.0
    q_drift_depth: float = 0.15    # slow random wobble on the band resonance
    q_drift_tau_s: float = 2.0

    # How much the level changes the CHARACTER of the wind, not only its loudness:
    # soft = muffled breeze, loud = bright gusty storm. 0 = level only.
    storm: float = 0.7
    drive: float = 1.2             # tanh saturation: 1.2 clean, >3 warm, >8 harsh
    # Left/right get two noises correlated by 1 - width. Mono-safe (no comb
    # filtering). 0 = dual mono, 1 = fully wide. The organ tone stays centred.
    stereo_width: float = 0.6

    # --- bourdon: narrow bandpass whistles on the same noise, at the pitch ---
    tone_level: float = 0.0        # 0 = off
    bourdon_q: float = 12.0        # higher = more pitched, lower = more airy
    use_fifth: bool = False        # add a whistle at the fifth
    third_mode: int = 0            # 0 = no third, 1 = minor (6:5), 2 = major (5:4)

    # --- organ: a pipe drone that the wind blows through ---
    organ_mode: bool = False
    # Hold one fixed note (organ_note, MIDI number) instead of following the
    # pitch. When it follows, the whole pitch range is squeezed onto a few bass
    # semitones starting at organ_octave.
    organ_hold: bool = False
    organ_note: float = 38.0       # 38 = D2 (~73 Hz)
    organ_octave: float = -2.0
    organ_brightness: float = 0.35  # 0 = dark, 1 = full bright chorus
    organ_air: float = 0.45        # pitched breath in the tube (noise at the harmonics)
    organ_wind: float = 0.7        # how much the gusts swell / brighten / sharpen the organ
    organ_level: float = 0.55      # organ loudness next to the wind
    # The organ is a slow colour, not a lead voice: it only builds while the
    # level stays above organ_threshold, over organ_rise_s, and fades over
    # organ_fall_s.
    organ_threshold: float = 0.55
    organ_rise_s: float = 6.0
    organ_fall_s: float = 2.5
    trem_depth: float = 0.05       # organ tremulant on the level
    trem_rate: float = 5.5         # Hz (a classic tremulant is 5-6 Hz)
    trem_pitch: float = 0.003      # tremulant on the pitch (fraction, ~5 cents)

    # --- room and output ---
    reverb_mix: float = 0.25       # 0 = dry (reverb off, no CPU)
    reverb_room: float = 0.7       # 0 = small room, 1 = long cathedral
    reverb_damping: float = 0.5    # 0 = bright tail, 1 = dark tail
    reverb_bass: float = 0.0       # lows ring longer (warm stone room). 0 = neutral
    wind_gain: float = 1.0         # master gain after the reverb
    # Speaker protection: a 24 dB/oct high-pass that removes the sub-bass you
    # can't hear but that shakes the woofer. The wind body sits at ~110 Hz.
    subsonic_hp_hz: float = 30.0

    # Play only these layers: "low", "mid", "high", "bourdon", "organ".
    # Empty = all of them. Replace the set, don't mutate it (the engine may be
    # reading it from the audio thread).
    solo: frozenset = field(default_factory=frozenset)

    def snapshot(self) -> "Params":
        """A copy for one audio block. Take it under the host's lock."""
        return copy.copy(self)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["solo"] = sorted(self.solo)
        return d

    def update(self, data: dict) -> None:
        """Set the known keys from data, skip the others, keep each field's type."""
        for f in fields(self):
            if f.name not in data:
                continue
            v = data[f.name]
            if f.name == "solo":
                v = frozenset(v)
            elif isinstance(getattr(self, f.name), bool):
                v = bool(v)
            elif isinstance(getattr(self, f.name), int):
                v = int(round(v))
            else:
                v = float(v)
            setattr(self, f.name, v)

    @classmethod
    def load(cls, path: str | Path) -> "Params":
        p = cls()
        p.update(json.loads(Path(path).read_text()))
        return p

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")
