"""Simple control helpers for a host: glide the controls, or play by itself.

The engine takes the pitch and level as they are, per block. A host with jumpy
input (keys, a sensor) runs them through a Smoother first. Wanderer makes slow
random wind gestures, for a demo or an installation with nobody playing.
"""
import math
import random

from .engine import SR, pitch_to_freq

SMOOTH_MS = 12.0   # pitch glide, and level fall (s/1000)
LEVEL_RISE_S = 0.6  # level rise: the wind swells slowly, it never blasts in


class Smoother:
    """One-pole glide on pitch and level, one step per audio block. The level
    rises slowly (rise_s) and falls fast (SMOOTH_MS), like a wind machine."""

    def __init__(self, sr: int = SR, rise_s: float = LEVEL_RISE_S):
        self.sr = sr
        self.rise_s = rise_s
        self.freq = pitch_to_freq(0.5)
        self.level = 0.0

    def step(self, frames: int, target_freq: float, target_level: float) -> tuple[float, float]:
        dt = frames / self.sr
        a_fast = 1.0 - math.exp(-dt / (SMOOTH_MS / 1000.0))
        a_rise = 1.0 - math.exp(-dt / max(self.rise_s, 1e-4))
        self.freq += (target_freq - self.freq) * a_fast
        a = a_rise if target_level > self.level else a_fast
        self.level += (target_level - self.level) * a
        return self.freq, self.level


# Wanderer gestures. pitch and level are 0..1 positions, like a hand on a theremin.
MIN_GESTURE_S = 1.2  # shortest glide from one waypoint to the next
MAX_GESTURE_S = 4.5  # longest glide (slow, sustained wind)
PITCH_STEP = 0.45    # max pitch change per gesture
LEVEL_LO = 0.45      # quietest "playing" level
LEVEL_HI = 1.0       # loudest gust
REST_PROB = 0.2      # chance a gesture is a near-silent lull instead
REST_LEVEL = 0.08    # level during a lull (the wind eases down, not fully out)


class Wanderer:
    """Slow random wind gestures: eases between random waypoints, with lulls.
    step(dt) returns (pitch, level), both 0..1. Pure: no thread, no clock."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.x0 = self.tx = self.rng.random()
        self.y0 = self.ty = LEVEL_LO
        self.glide = 1.0
        self.speed = 0.0

    def step(self, dt: float) -> tuple[float, float]:
        rng = self.rng
        if self.glide >= 1.0:
            self.x0, self.y0 = self.tx, self.ty
            self.tx = max(0.0, min(1.0, self.tx + rng.uniform(-PITCH_STEP, PITCH_STEP)))
            if rng.random() < REST_PROB:
                self.ty = rng.uniform(0.0, REST_LEVEL)
            else:
                self.ty = rng.uniform(LEVEL_LO, LEVEL_HI)
            self.speed = 1.0 / rng.uniform(MIN_GESTURE_S, MAX_GESTURE_S)
            self.glide = 0.0
        self.glide = min(1.0, self.glide + dt * self.speed)
        e = self.glide * self.glide * (3.0 - 2.0 * self.glide)  # smoothstep ease in/out
        return (self.x0 + (self.tx - self.x0) * e,
                self.y0 + (self.ty - self.y0) * e)
