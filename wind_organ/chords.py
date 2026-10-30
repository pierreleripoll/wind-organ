"""Just-intonation chords for the organ, and a player that steps through them.

A chord is text: ratios over the organ root, "1 3/2 7/4". Each ratio is one pipe.
In just intonation the notes are pure ratios of the harmonic series, so a chord
melts into one big tone instead of beating like equal temperament.

CHORDS and PIECE come from Ellen Arkbro, "For organ and brass" (2017). They are
measured from the recording, not read from a score: the strongest peaks of each
10 s block, snapped to the closest 7-limit ratio over C3 (organ_note 48), then
matched to the closest chord below. Octave doublings may be a bit off; the
harmony (the 7/4 Bb, the 7/6 Eb, the 21/20 C#) is the one of the piece.
"""
import random
from fractions import Fraction

MAX_VOICES = 16  # more pipes than this in one chord are dropped (CPU)

# Ratios over C3. Names are the chord over C, roughly.
CHORDS = {
    "c_high":    "2 7/2 4 6 7",                         # C Bb C G Bb, no bass
    "f_high":    "4/3 7/3 8/3 4 14/3 7",                # F Eb F C Eb Bb
    "c7":        "1 3/2 2 7/2 4 7",                     # C G C Bb C Bb (harmonics of C2)
    "c7_third":  "1 3/2 2 4 5 7",                       # with the pure 5/4 E
    "c_low":     "1/2 1 3/2 2 7/2 4",                   # with the C2 pedal
    "f_over_c":  "1 4/3 3/2 7/3 8/3 4 14/3",
    "f_pedal":   "2/3 1 4/3 3/2 4 14/3",                # F2 pedal
    "ab":        "14/9 8/3 28/9 14/3 16/3 8",
    "ab_over_c": "1 14/9 7/4 2 28/9 4 14/3",
    "ab_bass":   "8/9 14/9 16/9 2 28/9 4 16/3",         # Bb2 in the bass
    "eb":        "1 7/6 7/3",                           # the 7/6 septimal minor third
    "eb_f":      "2/3 7/6 4/3 8/3 4",
    "db":        "21/20 4/3 21/10 8/3 4",               # 21/20 C#, 20 cents flat
    "d_over_c":  "1/2 1 7/4 9/4 9/2 27/4",
    "eb_low":    "7/12 7/6 4/3 7/4 8/3 4 14/3 16/3",    # Eb2 pedal
    "c_sept":    "1/2 1 7/6 4/3 3/2 7/4 7/2 14/3 21/4 7",
}

# The chords of the piece in order, one per 10 s (20 min).
PIECE = """
c_high f_high c_high f_high c7 f_over_c c7 f_over_c c7 f_over_c ab eb
c7 f_over_c c7_third f_over_c c7 f_over_c ab eb ab_over_c f_over_c ab ab_over_c
c7 f_pedal c7 f_over_c c7 f_pedal ab eb c7_third f_pedal c_low f_over_c
c7 f_pedal ab eb ab_over_c f_pedal f_pedal eb_f db db eb_f db
eb_f f_over_c eb_f f_over_c db eb_f db eb_f db eb_f f_pedal eb_f
f_over_c db db eb_f ab_over_c
d_over_c c_sept eb_low d_over_c c_sept eb_low d_over_c c_sept eb_low
d_over_c c_sept eb_low d_over_c c_sept eb_low d_over_c c_sept eb_low
d_over_c c_sept eb_low
c7 f_pedal c_low f_pedal c_low f_over_c ab_bass eb_f c7 f_pedal c_low f_over_c
c_low f_pedal ab_bass eb c7 f_over_c c7 f_over_c c7 f_over_c ab eb
c_high f_high c_high f_high c_high f_high ab eb eb eb
""".split()
PIECE_STEP_S = 10.0


def parse_chord(text: str) -> tuple[Fraction, ...]:
    """'1 3/2 7/4' -> (1, 3/2, 7/4), sorted, no doubles. '' -> (1,), one pipe at
    the root. Raises ValueError on a bad ratio."""
    ratios = sorted({Fraction(t) for t in text.replace(",", " ").split()})
    if any(r <= 0 for r in ratios):
        raise ValueError(f"chord ratios must be > 0: {text!r}")
    return tuple(ratios[:MAX_VOICES]) or (Fraction(1),)


class ChordPlayer:
    """Plays a chord progression by itself, one chord every step_s seconds.

    mode "piece" plays the names in order and loops. mode "wander" picks each
    next chord at random among the chords that follow it in the sequence, so it
    keeps the moves of the piece but never the same path. step(dt) returns the
    chord text for Params.organ_chord. start is the index of the first chord
    (in the piece, chord i starts at i * 10 s). Pure: no thread, no clock.

    With ramp_s > 0 the chords start slow and speed up: the first chord lasts
    slow_s, then the length goes down (exponential curve) to step_s at ramp_s
    seconds. restart() starts the ramp again."""

    def __init__(self, sequence: list[str] | None = None, step_s: float = PIECE_STEP_S,
                 mode: str = "piece", rng: random.Random | None = None, start: int = 0,
                 slow_s: float | None = None, ramp_s: float = 0.0):
        if mode not in ("piece", "wander"):
            raise ValueError(f"unknown chord mode {mode!r}")
        self.sequence = list(sequence or PIECE)
        self.step_s = step_s
        self.mode = mode
        self.rng = rng or random.Random()
        self.next_of: dict[str, list[str]] = {}
        for a, b in zip(self.sequence, self.sequence[1:] + self.sequence[:1]):
            if a != b:  # a chord that repeats would just hold the walk in place
                self.next_of.setdefault(a, []).append(b)
        self.index = start % len(self.sequence)
        self.name = self.sequence[self.index]
        self.t = 0.0
        self.slow_s = slow_s or step_s
        self.ramp_s = ramp_s
        self.elapsed = 0.0  # seconds since the chords started, for the ramp

    def restart(self) -> None:
        self.t = self.elapsed = 0.0

    def step_now(self) -> float:
        """Length of a chord at this point of the ramp (s)."""
        if self.ramp_s <= 0.0 or self.elapsed >= self.ramp_s:
            return self.step_s
        return self.slow_s * (self.step_s / self.slow_s) ** (self.elapsed / self.ramp_s)

    def step(self, dt: float) -> str:
        self.t += dt
        self.elapsed += dt
        while self.t >= self.step_now():
            self.t -= self.step_now()
            if self.mode == "piece":
                self.index = (self.index + 1) % len(self.sequence)
                self.name = self.sequence[self.index]
            else:
                self.name = self.rng.choice(self.next_of.get(self.name, self.sequence))
        return CHORDS[self.name]
