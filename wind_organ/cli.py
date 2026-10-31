"""Play the wind + organ from the keyboard, let it play by itself, or render a WAV.

    python -m wind_organ                       # TUI: arrows play, space = autoplay
    python -m wind_organ --no-tui              # autoplay only, Ctrl+C to stop
    python -m wind_organ --render out.wav --seconds 30
    python -m wind_organ --preset preset.json --organ
    python -m wind_organ --preset presets/organ_and_brass.json --chords piece --floor 0.4
    python -m wind_organ --preset presets/slow_organ.json --chords wander --chords-after 180 --floor 0.4
    python -m wind_organ --preset presets/slow_organ.json --chords wander --cue --floor 0.4
    python -m wind_organ --list                # audio devices
"""
import argparse
import random
import sys
import threading

import numpy as np

from .chords import ChordPlayer
from .controls import Smoother, Wanderer
from .convolver import find_ir
from .engine import SR, WindOrgan, pitch_to_freq
from .params import Params

BLOCK = 512        # ~10.7 ms per audio callback
LEVEL_CURVE = 3.0  # level^curve: soft levels get most of the key/gesture travel


class Player:
    """What the UI thread and the audio thread share, guarded by `lock`."""

    def __init__(self, params: Params, autoplay: bool, seed: int | None = None,
                 floor: float = 0.0, chords: ChordPlayer | None = None,
                 chords_after: float = 0.0):
        self.lock = threading.Lock()
        self.params = params
        self.pitch = 0.4   # 0..1
        self.level = 0.0   # 0..1, before LEVEL_CURVE
        self.autoplay = autoplay
        self.synth = WindOrgan(SR, seed=seed)
        self.smoother = Smoother(SR)
        self.wanderer = Wanderer(random.Random(seed), floor=floor)
        # Chord autoplay: steps Params.organ_chord through a progression.
        self.chords = chords
        self.chord_auto = chords is not None
        # Seconds of the organ's own note before the chords start (--chords-after).
        self.chord_wait = chords_after

    def block(self, frames: int) -> np.ndarray:
        with self.lock:
            if self.autoplay:
                self.pitch, self.level = self.wanderer.step(frames / SR)
            if self.chord_auto:
                if self.chord_wait > 0.0:
                    self.chord_wait -= frames / SR
                else:
                    self.params.organ_chord = self.chords.step(frames / SR)
            pitch, level = self.pitch, self.level
            p = self.params.snapshot()
            if self.chord_auto and self.chord_wait <= 0.0:
                # Slow chords glide slowly: the crossfade follows the chord length.
                p.organ_fade_s *= self.chords.step_now() / self.chords.step_s
        freq, lvl = self.smoother.step(frames, pitch_to_freq(pitch), level ** LEVEL_CURVE)
        return self.synth.render(frames, freq, lvl, p)


def run_stream(player: Player, device, body) -> None:
    import sounddevice as sd

    def callback(outdata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        out = player.block(frames)
        if outdata.shape[1] == 2:
            outdata[:] = out.astype(np.float32)
        else:
            outdata[:] = out.mean(axis=1).astype(np.float32)[:, None]

    with sd.OutputStream(samplerate=SR, blocksize=BLOCK, channels=2, dtype="float32",
                         device=device, callback=callback):
        body()


def render_wav(player: Player, path: str, seconds: float) -> None:
    from scipy.io import wavfile

    n = int(seconds * SR / BLOCK)
    out = np.concatenate([player.block(BLOCK) for _ in range(n)], axis=0)
    wavfile.write(path, SR, (out * 32767.0).astype(np.int16))
    print(f"wrote {path}: {n * BLOCK / SR:.1f} s, peak {np.abs(out).max():.2f}")


def tui(stdscr, player: Player, preset_path: str | None) -> None:
    import curses

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    msg = ""
    while True:
        with player.lock:
            p = player.params
            pitch, level, auto = player.pitch, player.level, player.autoplay
            lines = [
                "--- wind organ ---",
                "",
                f"pitch {pitch_to_freq(pitch):7.1f} Hz  [{'#' * int(pitch * 30):<30}]",
                f"level {level:7.2f}     [{'#' * int(level * 30):<30}]",
                f"organ swell {player.synth.organ_swell:4.2f}"
                + (f"   held note {p.organ_note:.0f}" if p.organ_hold else ""),
                f"chord [{p.organ_chord or '1'}]"
                + (f" {player.chords.name} ({player.chords.step_now():.0f} s)"
                   if player.chord_auto and player.chord_wait <= 0.0 else "")
                + (f" chords in {player.chord_wait:.0f} s" if player.chord_auto and player.chord_wait > 0.0 else "")
                + f"   ranks {p.organ_ranks}",
                "",
                f"[{'x' if auto else ' '}] autoplay (space)   [{'x' if p.organ_mode else ' '}] organ (o)"
                f"   [{'x' if p.organ_hold else ' '}] hold note (h)",
                f"[{'x' if p.use_gust else ' '}] gust (g)   [{'x' if p.use_3band else ' '}] 3-band (3)"
                f"   bourdon {p.tone_level:.2f} (b/B)",
                f"[{'x' if p.organ_mode and p.organ_gate > 0.0 else ' '}] drone (d)"
                f"   [{'x' if player.chord_auto else ' '}] chord autoplay (c)   ranks (r)"
                f"   wind chord {p.wind_chord:.2f} (n/N)",
                f"organic {p.organ_organic:.1f} (x/X: clean .. wind)"
                f"   cavity {p.organ_cavity:.1f} (k/K: pipe .. crack)   whistles {p.whistles} (v)",
                "",
                "left/right pitch   up/down level   [ ] held note"
                + ("   w save preset" if preset_path else "") + "   q quit",
                msg,
            ]
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        for y, line in enumerate(lines[:h]):
            stdscr.addnstr(y, 0, line, w - 1)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), 27):
            return
        with player.lock:
            p = player.params
            if key in (curses.KEY_LEFT, curses.KEY_RIGHT, curses.KEY_UP, curses.KEY_DOWN):
                player.autoplay = False  # take over from where the gesture is
                if key == curses.KEY_LEFT:
                    player.pitch = max(0.0, player.pitch - 0.02)
                elif key == curses.KEY_RIGHT:
                    player.pitch = min(1.0, player.pitch + 0.02)
                elif key == curses.KEY_UP:
                    player.level = min(1.0, player.level + 0.05)
                else:
                    player.level = max(0.0, player.level - 0.05)
            elif key == ord(" "):
                player.autoplay = not player.autoplay
            elif key == ord("o"):
                p.organ_mode = not p.organ_mode
            elif key == ord("h"):
                p.organ_hold = not p.organ_hold
            elif key == ord("["):
                p.organ_note = max(12.0, p.organ_note - 1.0)
            elif key == ord("]"):
                p.organ_note = min(84.0, p.organ_note + 1.0)
            elif key == ord("g"):
                p.use_gust = not p.use_gust
            elif key == ord("3"):
                p.use_3band = not p.use_3band
            elif key == ord("b"):
                p.tone_level = max(0.0, round(p.tone_level - 0.05, 2))
            elif key == ord("B"):
                p.tone_level = min(1.0, round(p.tone_level + 0.05, 2))
            elif key == ord("d"):
                # The drone fades in over organ_rise_s and out over organ_fall_s.
                if p.organ_mode and p.organ_gate > 0.0:
                    p.organ_gate = 0.0
                else:
                    p.organ_mode = True
                    p.organ_gate = 1.0
            elif key == ord("c"):
                if player.chords is None:
                    player.chords = ChordPlayer()
                player.chord_auto = not player.chord_auto
                if player.chord_auto:
                    player.chord_wait = 0.0  # start now, even during --chords-after
                    player.chords.restart()  # slow again, if --chord-ramp
                    p.organ_mode = True
                    p.organ_gate = 1.0
                else:
                    p.organ_chord = ""  # back to the single note
            elif key == ord("n"):
                p.wind_chord = max(0.0, round(p.wind_chord - 0.05, 2))
            elif key == ord("N"):
                p.wind_chord = min(1.0, round(p.wind_chord + 0.05, 2))
            elif key == ord("r"):
                p.organ_ranks = p.organ_ranks % 3 + 1
            elif key == ord("x"):
                p.organ_organic = max(0.0, round(p.organ_organic - 0.1, 1))
            elif key == ord("X"):
                p.organ_organic = min(1.0, round(p.organ_organic + 0.1, 1))
            elif key == ord("k"):
                p.organ_cavity = max(0.0, round(p.organ_cavity - 0.1, 1))
            elif key == ord("K"):
                p.organ_cavity = min(1.0, round(p.organ_cavity + 0.1, 1))
            elif key == ord("v"):
                p.whistles = (p.whistles + 1) % 5
            elif key == ord("w") and preset_path:
                p.save(preset_path)
                msg = f"saved {preset_path}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="wind-organ", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", help="JSON file with knob values (unknown keys are skipped)")
    ap.add_argument("--organ", action="store_true", help="start with the organ on")
    ap.add_argument("--hold", type=float, metavar="NOTE",
                    help="organ holds this MIDI note (implies --organ)")
    ap.add_argument("--chords", choices=["piece", "wander"],
                    help="chord autoplay: the chords of 'For organ and brass' in order, "
                         "or a random walk on the same moves (implies --organ, holds C3, "
                         "starts the wind autoplay)")
    ap.add_argument("--chord-s", type=float, default=10.0, help="seconds per chord")
    ap.add_argument("--chords-after", type=float, default=0.0, metavar="SECONDS",
                    help="hold the organ's single note this long, then start the chords")
    ap.add_argument("--chord-ramp", type=float, default=0.0, metavar="SECONDS",
                    help="start the chords slow and speed up to --chord-s over this time")
    ap.add_argument("--chord-slow-s", type=float, default=40.0,
                    help="length of the first chord with --chord-ramp")
    ap.add_argument("--chord-start", type=float, default=0.0, metavar="SECONDS",
                    help="start the chords at this time of the piece (650 = the C2 pedal part)")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="autoplay: quietest wind level, 0..1 (0 = lulls go near silent); "
                         "starts the wind autoplay")
    ap.add_argument("--cue", action="store_true",
                    help="start with the wind only: d brings the organ note in, "
                         "c starts the chords (keyboard UI)")
    ap.add_argument("--ir", metavar="WAV",
                    help="reverb in a real space: an impulse response (a WAV path, or a "
                         "file name of wind_organ/irs/). 'none' = the built room")
    ap.add_argument("--no-tui", action="store_true", help="no keyboard UI, autoplay only")
    ap.add_argument("--render", metavar="WAV", help="render offline to a WAV file (autoplay)")
    ap.add_argument("--seconds", type=float, default=30.0, help="length for --render")
    ap.add_argument("--seed", type=int, help="random seed (same seed = same render)")
    ap.add_argument("--device", help="output device (index or name, see --list)")
    ap.add_argument("--list", action="store_true", help="list audio devices and exit")
    args = ap.parse_args(argv)

    if args.list:
        import sounddevice as sd
        print(sd.query_devices())
        return

    params = Params.load(args.preset) if args.preset else Params()
    if args.ir:
        params.reverb_ir = "" if args.ir == "none" else args.ir
    if params.reverb_ir:
        try:
            find_ir(params.reverb_ir)
        except FileNotFoundError as e:
            ap.error(str(e))
    if args.organ or args.hold is not None:
        params.organ_mode = True
    if args.hold is not None:
        params.organ_hold = True
        params.organ_note = args.hold
    chords = None
    if args.cue and not args.chords:
        args.chords = "piece"
    if args.chords:
        params.organ_mode = True
        if args.hold is None:
            params.organ_hold = True
            params.organ_note = 48.0  # C3: the chords are ratios over it
        chords = ChordPlayer(step_s=args.chord_s, mode=args.chords, rng=random.Random(args.seed),
                             slow_s=args.chord_slow_s, ramp_s=args.chord_ramp,
                             start=int(args.chord_start // 10.0))

    # --chords and --floor are autoplay settings, so they start the wind autoplay
    # in the keyboard UI too (space still stops it, the arrows take over).
    autoplay = bool(args.render or args.no_tui or args.chords or args.floor > 0.0)
    player = Player(params, autoplay=autoplay, seed=args.seed, floor=args.floor, chords=chords,
                    chords_after=args.chords_after)
    if args.cue:
        # Wind only. The organ is ready but silent, the chords wait for c.
        params.organ_gate = 0.0
        player.chord_auto = False
    if args.render:
        render_wav(player, args.render, args.seconds)
        return

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    if args.no_tui:
        stop = threading.Event()

        def wait():
            print("playing by itself, Ctrl+C to stop")
            try:
                stop.wait()
            except KeyboardInterrupt:
                pass

        run_stream(player, device, wait)
    else:
        import curses
        run_stream(player, device, lambda: curses.wrapper(tui, player, args.preset))


if __name__ == "__main__":
    main()
