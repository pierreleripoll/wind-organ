"""Wind + pipe-organ synth engine.

    from wind_organ import Params, WindOrgan, pitch_to_freq

    p = Params(organ_mode=True)
    synth = WindOrgan()
    block = synth.render(512, pitch_to_freq(0.3), 0.7, p)   # (512, 2) float64
"""
from .controls import Smoother, Wanderer
from .engine import FREQ_HI, FREQ_LO, SR, WindOrgan, pitch_to_freq
from .params import Params
from .reverb import StereoReverb

__all__ = ["FREQ_HI", "FREQ_LO", "SR", "Params", "Smoother", "StereoReverb",
           "Wanderer", "WindOrgan", "pitch_to_freq"]
