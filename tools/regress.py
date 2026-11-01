"""Check that a change does not change the sound by accident.

It renders a few seeded scenes (same seed = same sound) with the code of a git
revision, and with the code of the working tree, and compares them sample by
sample. A new knob that is off by default must give "same" everywhere.

    .venv/bin/python tools/regress.py              # working tree against HEAD
    .venv/bin/python tools/regress.py main~3       # against another revision
    .venv/bin/python tools/regress.py --wav out/   # also write the new renders as WAV

The exit code is 1 when a scene changed. The presets come from each side too,
so a changed preset shows as a changed scene.
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED = 3
FLOOR = 0.4

# name: (preset file or None, chord autoplay or None, organ on)
SCENES = {
    "wind": (None, None, False),
    "organ": (None, None, True),
    "brass": ("organ_and_brass.json", "piece", True),
    "inwind": ("chords_in_wind.json", "wander", True),
    "slow": ("slow_organ.json", None, True),
    "cave": ("cave.json", None, False),
    "ruin": ("ruin.json", "piece", False),
}


def render_all(out: str, seconds: float) -> None:
    """Render every scene with the wind_organ found on the path (one side)."""
    import wind_organ
    from wind_organ.chords import ChordPlayer
    from wind_organ.cli import BLOCK, Player
    from wind_organ.engine import SR
    from wind_organ.params import Params

    print(f"  rendering with {Path(wind_organ.__file__).parent}", file=sys.stderr)
    arrays = {}
    for name, (preset, chords, organ) in SCENES.items():
        path = Path("presets") / preset if preset else None
        if path is not None and not path.is_file():
            continue
        p = Params.load(path) if path else Params()
        if organ:
            p.organ_mode = True
        player_chords = None
        if chords:
            p.organ_mode, p.organ_hold, p.organ_note = True, True, 48.0
            player_chords = ChordPlayer(step_s=10.0, mode=chords, rng=random.Random(SEED))
        player = Player(p, autoplay=True, seed=SEED, floor=FLOOR, chords=player_chords)
        n = int(seconds * SR / BLOCK)
        arrays[name] = np.concatenate([player.block(BLOCK) for _ in range(n)], axis=0)
    np.savez(out, **arrays)


def run_side(src: Path, out: Path, seconds: float) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=str(src))
    return subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                             "--render-to", str(out), "--seconds", str(seconds)],
                            cwd=src, env=env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rev", nargs="?", default="HEAD", help="git revision to compare with")
    ap.add_argument("--seconds", type=float, default=20.0, help="length of each scene")
    ap.add_argument("--wav", metavar="DIR", help="also write the new renders as WAV here")
    ap.add_argument("--render-to", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.render_to:
        render_all(args.render_to, args.seconds)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        old_src = tmp / "old"
        old_src.mkdir()
        archive = subprocess.run(["git", "archive", args.rev, "wind_organ", "presets"],
                                 cwd=ROOT, check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(old_src)], input=archive, check=True)
        print(f"rendering {len(SCENES)} scenes of {args.seconds:g} s, {args.rev} and the working tree",
              file=sys.stderr)
        sides = [run_side(old_src, tmp / "old.npz", args.seconds),
                 run_side(ROOT, tmp / "new.npz", args.seconds)]
        if any(s.wait() != 0 for s in sides):
            print("a render failed", file=sys.stderr)
            return 2
        old, new = np.load(tmp / "old.npz"), np.load(tmp / "new.npz")

        changed = 0
        for name in SCENES:
            if name not in old or name not in new:
                print(f"{name:7s} skipped (missing on one side)")
                continue
            a, b = old[name], new[name]
            if a.shape != b.shape:
                print(f"{name:7s} CHANGED (shape {a.shape} -> {b.shape})")
                changed += 1
                continue
            diff = float(np.abs(a - b).max())
            if diff == 0.0:
                print(f"{name:7s} same")
                continue
            rel = 20 * np.log10(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a ** 2)))
            print(f"{name:7s} CHANGED (max diff {diff:.2e}, diff/signal {rel:.1f} dB)")
            changed += 1

        if args.wav:
            from scipy.io import wavfile

            from wind_organ.engine import SR
            out = Path(args.wav)
            out.mkdir(parents=True, exist_ok=True)
            for name in new.files:
                wavfile.write(out / f"{name}.wav", SR, (new[name] * 32767).astype(np.int16))
            print(f"wrote {len(new.files)} WAV files in {out}")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
