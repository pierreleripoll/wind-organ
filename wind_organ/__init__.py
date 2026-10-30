"""Wind + pipe-organ synth engine.

    from wind_organ import Params, WindOrgan, pitch_to_freq

    p = Params(organ_mode=True)
    synth = WindOrgan()
    block = synth.render(512, pitch_to_freq(0.3), 0.7, p)   # (512, 2) float64
"""
from .chords import CHORDS, PIECE, ChordPlayer, parse_chord
from .controls import Smoother, Wanderer
from .engine import FREQ_HI, FREQ_LO, SR, WindOrgan, pitch_to_freq
from .params import Params
from .reverb import StereoReverb

__all__ = ["CHORDS", "FREQ_HI", "FREQ_LO", "PIECE", "SR", "ChordPlayer", "Params",
           "Smoother", "StereoReverb", "Wanderer", "WindOrgan", "parse_chord",
           "pitch_to_freq"]
