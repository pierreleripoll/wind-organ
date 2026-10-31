# wind organ

A real-time synth that makes wind, and a pipe organ that the wind blows through.

This is the sound engine of the [theremin of winds](https://github.com/pierreleripoll/theremin-of-winds), without the theremin. It has no input device. You give it two numbers per audio block, a pitch and a level, and it gives you stereo audio. The sound is built from scratch: no samples, no recordings.

## Install

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[play]'
```

The `play` extra adds `sounddevice`, which the command line player needs. The engine alone needs only `numpy`, `scipy` and `numba` (numba compiles the loop of the blown pipes, see `tube.py`).

## Play

```bash
.venv/bin/python -m wind_organ                      # keyboard UI
.venv/bin/python -m wind_organ --organ              # start with the organ on
.venv/bin/python -m wind_organ --hold 38            # organ holds one note (MIDI 38 = D2)
.venv/bin/python -m wind_organ --no-tui             # plays by itself, Ctrl+C to stop
.venv/bin/python -m wind_organ --render out.wav --seconds 30 --seed 1
.venv/bin/python -m wind_organ --preset preset.json # load knob values from a JSON file
.venv/bin/python -m wind_organ --preset presets/chords_in_wind.json --chords piece --floor 0.4
.venv/bin/python -m wind_organ --preset presets/cave.json --floor 0.4   # wind that finds a pipe
.venv/bin/python -m wind_organ --list               # list the audio devices
```

In the keyboard UI:

| Key | Action |
|---|---|
| left / right | Pitch down / up |
| up / down | Level up / down |
| space | Autoplay on / off (slow random wind gestures) |
| `o` | Organ on / off |
| `h` | Organ holds one note / follows the pitch |
| `[` `]` | Held note down / up |
| `g` | Gusts on / off |
| `3` | 3-band wind / single band |
| `b` `B` | Bourdon level down / up |
| `d` | Drone on / off: the organ fades in over `organ_rise_s`, out over `organ_fall_s` |
| `c` | Chord autoplay on / off (the chords of the piece, see below). Off = back to the single note |
| `r` | Ranks per pipe: 1, 2, 3 |
| `n` `N` | Wind chord level down / up |
| `x` `X` | `organ_organic` down / up: from a clean organ to wind that finds a pipe (see below) |
| `k` `K` | `organ_cavity` down / up: the blown tubes go from a straight pipe to a crack in the rock |
| `v` | Edge whistles: 0, 1, 2, 3, 4 |
| `w` | Save the knobs to the `--preset` file |
| `q` | Quit |

## Chords, like "For organ and brass"

The organ can play a chord instead of one note. `organ_chord` is text: ratios over the organ root, like `"1 3/2 7/4"`. Each ratio is one pipe. The ratios are just intonation (pure ratios of the harmonic series), so the notes melt into one tone. Empty text means one pipe at the root, like before.

[`wind_organ/chords.py`](wind_organ/chords.py) has the chords of Ellen Arkbro, *For organ and brass* (2017). They are measured from the recording, not read from a score: one chord per 10 s, as 7-limit ratios over C3. The piece uses the 7/4 Bb (31 cents flat), the 7/6 Eb and the 21/20 C#.

The chord can sound in two ways, and you can mix them:

- `wind_chord`: the wind noise itself rings at the chord notes (narrow bandpass filters). It follows the wind level and the gusts, so the chord is half heard, inside the wind. It works with the organ off.
- The organ: sine pipes at the chord notes, a clear tone.

Two presets:

- [`presets/chords_in_wind.json`](presets/chords_in_wind.json): the wind first. The chord is about 11 dB under the wind and comes up with the gusts. The organ is very low and only comes up in strong wind.
- [`presets/organ_and_brass.json`](presets/organ_and_brass.json): the organ first, as close to the record as possible. The wind is only a colour on it.

```bash
# the chords in the order of the piece, inside a wind that never stops
.venv/bin/python -m wind_organ --preset presets/chords_in_wind.json --chords piece --floor 0.4
# start at the C2 pedal part (10:50 in the piece)
.venv/bin/python -m wind_organ --preset presets/chords_in_wind.json --chords piece --chord-start 650
# a random walk on the same chord moves, 15 s per chord
.venv/bin/python -m wind_organ --preset presets/chords_in_wind.json --chords wander --chord-s 15
# the organ in front, like the record
.venv/bin/python -m wind_organ --preset presets/organ_and_brass.json --chords piece --floor 0.4
# one organ note that comes up slowly, then the chords after 3 min
.venv/bin/python -m wind_organ --preset presets/slow_organ.json --chords wander --chords-after 180 --floor 0.4
```

To play it by hand, use `--cue`: it starts with the wind only. `d` brings the organ note in, slowly. `c` starts the chords when you want, and stops them (back to the single note). `d` again fades the organ out.

```bash
.venv/bin/python -m wind_organ --preset presets/slow_organ.json --chords wander --cue --chord-ramp 180 --floor 0.4
```

`--chords-after` holds the organ on its single note (the root, C3) for that many seconds, then starts the chords. With [`presets/slow_organ.json`](presets/slow_organ.json) the organ comes up in about 1 min (`organ_rise_s` 20). It follows the wind a bit (`organ_follow` 0.4, `organ_wind` 1): louder in strong wind, softer in the lulls, about 16 dB between the two.

`--chord-ramp SECONDS` starts the chords slow and speeds them up: the first chord lasts `--chord-slow-s` (40 s by default), then the chords get shorter, down to `--chord-s` at the end of the ramp. The crossfade follows the chord length, so the first chords glide slowly into each other. In `--cue`, each `c` that starts the chords starts the ramp again.

`--chords` turns the organ on, holds C3 (unless you give `--hold`) and starts the wind autoplay, also in the keyboard UI. Space stops it, the arrows take over. `--floor` sets the quietest wind level in autoplay (0 to 1): the wind still changes colour and strength, but it never goes silent.

The new knobs (the "Preset" column is `organ_and_brass.json`):

| Knob | Preset | Default | What it does |
|---|---|---|---|
| `wind_chord` | 0 (0.35 in `chords_in_wind`) | 0 (off) | The wind rings at the chord notes. |
| `wind_chord_q` | 40 (30 in `chords_in_wind`) | 40 | Higher = more pitched, lower = more airy. |
| `organ_ranks` | 3 | 1 | Copies of each pipe, a bit detuned. They beat slowly, like real pipes in a room. |
| `organ_beat_hz` | 0.35 | 0.3 | Widest detune between ranks (Hz). The beat speed drifts around it. |
| `organ_spread` | 1.0 | 0.0 | Spreads pipes and ranks left and right, with a small phase offset per rank (like spaced mics). Mono loses 3 dB at most. |
| `organ_tone_hz` | 1200 | 0 (off) | Softens the partials above this frequency, so high pipes stay dark, like soft brass. |
| `organ_harm7` | 1.0 | 0.0 | Adds the 7th harmonic to each pipe. |
| `organ_gate` | 1.0 | 1.0 | 1 = the organ plays. 0 = it fades out over `organ_fall_s` (the `d` key sets it). |
| `organ_follow` | 0.0 | 1.0 | 1 = the organ only swells with strong wind. 0 = a steady drone, and the gusts only colour it. |
| `organ_fade_s` | 1.0 | 1.0 | Crossfade time when the chord changes. |

With the default values, the engine gives the same sound as before.

The preset file is plain JSON. The loader skips keys that it does not know, so a `preset.json` from the theremin of winds loads as it is.

## A wind that finds a pipe

`organ_organic` goes from a clean organ (0) to a natural accident (1): the wind blows into a crack of a cave or a ruin, and it sounds like an organ pipe. You can move it during the piece (`x` and `X` in the keyboard UI).

On the way, the clean sine pipes get less clean:

- Each partial moves slowly on its own, like a real pipe in moving air.
- The air pressure decides how each pipe speaks. The pressure is the organ swell times the gust. With weak air the fundamental is gone and the octave flickers. With a strong gust the pipe overblows to the octave. Each pipe speaks at a slightly different pressure.
- The pitch shimmers a few cents, more while the pipe is unstable.
- A pipe that starts to speak gives a short puff of noise (the chiff).

At the same time, each sine pipe crossfades into a blown tube ([`wind_organ/tube.py`](wind_organ/tube.py)). The tube is a small physical model of a flue pipe: an air column, and a jet of air that hits an edge. The jet needs time to cross the mouth, and the pressure sets this time. So the pressure picks the note by itself, with no rule written by hand:

- With weak air the tube only breathes. Just before it catches, it can whisper the octave.
- With normal air it sings, in tune within about 10 cents. It goes a bit sharp when the air pushes harder, like a real flute. Low pipes are slow to speak.
- In a big gust it jumps to the octave. It stays there until the gust is well gone.

Its tone is mostly odd harmonics, like a stopped organ pipe (a bourdon). The wind noise shakes the jet all the time, so the tone breathes. So at 1, the organ is made by the wind itself.

`organ_cavity` (keys `k` and `K`) bends the tubes into a crack in the rock. A real cavity is not a straight pipe: the modes above the note are not harmonics. The note stays in tune, but the tone gets purer and hollower, like wind across a bottle. The breath rings at odd pitches. A gust jumps to a strange interval: at 1 it is near a ninth (2.25 times the note), not the octave. Each pipe bends a bit differently.

Four knobs make the wind itself more real. They are all off by default:

| Knob | Try | What it does |
|---|---|---|
| `gust_bright` | 0.6 to 1 | A gust brightens the wind: the mid and high bands follow it more than the low rumble. |
| `buffet_depth` | 0.3 to 0.5 | The fast flutter of real wind (`buffet_hz`, 12 Hz by default). All bands flap together. |
| `gust_sweep_s` | 0.5 to 1 | A gust reaches one side first, the other side up to this many seconds later. The wind direction drifts slowly. |
| `whistles` | 1 to 4 | Edge whistles: the wind on a wire or a crack. Each one has its own pitch, sings only above its own wind speed, and goes higher when the wind blows harder. `whistle_level`, `whistle_q`, `whistle_track` shape them. |

Two presets use all of this:

- [`presets/cave.json`](presets/cave.json): wind in a crack finds a low pipe and its fifth. Only blown tubes, bent into a crack (`organ_cavity` 1): hollow tones, and a big gust jumps to a ninth. Slow big gusts, a long dark room, one far whistle. The pipes breathe, catch and sing in turn.
- [`presets/ruin.json`](presets/ruin.json): a storm in a ruin. Three pipes, mostly blown (`organ_organic` 0.7) and a bit crooked (`organ_cavity` 0.3), driven by the wind. Bright gusts that sweep across, flutter, three whistles.

```bash
.venv/bin/python -m wind_organ --preset presets/cave.json --floor 0.4
.venv/bin/python -m wind_organ --preset presets/ruin.json --floor 0.4
# the chords of the piece, played by blown tubes
.venv/bin/python -m wind_organ --preset presets/ruin.json --chords piece --floor 0.4
```

With all these knobs at 0, the engine gives the same sound as before. Only two small changes are always on: the wind level and the organ level and pitch now glide inside each block instead of moving in steps.

## A real space

`reverb_ir` replaces the built room with a real space. You give it an impulse response (IR): the recording of one click in a cave, a church or a mine. The IR holds all the echoes of that space. The engine plays the sound through it (a convolution, see [`wind_organ/convolver.py`](wind_organ/convolver.py)).

```bash
.venv/bin/python -m wind_organ --preset presets/cave.json --floor 0.4 --ir ~/irs/my_cave.wav   # any WAV
```

- The IR is a mono or stereo WAV, at any sample rate. Give a path, or the name of a file in `wind_organ/irs/`. In a preset, it is the `reverb_ir` key. `--ir none` goes back to the built room.
- The engine cuts the silence and the direct sound (the click itself) at the start of the IR. It sets the level so that `reverb_mix` sounds about as loud as with the built room.
- With an IR, `reverb_room`, `reverb_damping` and `reverb_bass` do nothing. The space is what it is.
- It costs about 0.15 ms per block for a 6 s stereo IR. It adds no delay with blocks of 512 samples.
- When `reverb_ir` changes, the engine reads the file in the audio thread. So the block after a change can click.

One IR comes with the engine: `es_culleram`, a small Punic sanctuary cave in Ibiza (the sound dies in about 1 s). It is not under the MIT license: it is CC BY-NC-SA 4.0, so no commercial use. The credits are in [`wind_organ/irs/README.md`](wind_organ/irs/README.md).

```bash
.venv/bin/python -m wind_organ --preset presets/ruin.json --floor 0.4 --ir es_culleram
```

## Use it as a library

```python
from wind_organ import Params, Smoother, WindOrgan, pitch_to_freq

p = Params(organ_mode=True, organ_hold=True, organ_note=38)
synth = WindOrgan()
smooth = Smoother()

# In your audio callback, once per block:
freq, level = smooth.step(512, pitch_to_freq(0.3), 0.7)
block = synth.render(512, freq, level, p)   # (512, 2) float64, in [-1, 1]
```

The engine is not thread-safe. Call it from the audio thread only. If another thread changes the knobs, keep the `Params` behind a lock and give the engine `p.snapshot()` for each block.

`render()` runs the full chain. If you must add something before the limiter (a microphone, for example), call the three steps yourself:

```python
sat = synth.voice(frames, freq, level, p)   # wind + bourdon + organ, saturated
out = synth.room(sat, p)                    # reverb, then wind_gain
out = out + my_mic_signal
out = synth.protect(out, p)                 # subsonic high-pass + limiter
```

To use it from another project, install it in that project's venv:

```bash
uv pip install --python .venv/bin/python -e ../wind-organ
```

## How the sound is made

1. Pink noise is the raw material. It has more energy in the lows than white noise, so it sounds darker and fuller.
2. Three bandpass filters shape it. A low band gives a fixed rumble (about 110 Hz). A mid band follows the pitch. A high whistle (about 3.2 kHz) comes in at high pitch.
3. The level sets the loudness. The `storm` knob also lets it change the character: a soft level gives a muffled breeze, a loud level gives a bright, gusty storm.
4. A slow random LFO (low-frequency oscillator, a slow control signal) swells the level. These are the gusts. With `gust_layers` above 1, faster gust layers add turbulence. A gust can also brighten the wind (`gust_bright`) and reach one side first (`gust_sweep_s`). The buffet adds a fast flutter on top (`buffet_depth`).
5. The bourdon adds narrow whistles at the pitch, the fifth and the third. It sounds like air across a bottle. The wind chord does the same at the notes of `organ_chord`.
6. The edge whistles (`whistles`) are narrow bandpasses on the same noise that come in only in strong wind, and rise with it.
7. The organ adds a pipe drone: sine partials at the harmonics of a deep note, plus noise that rings at the same harmonics (the air in the tube). The same gusts swell, brighten and sharpen it. It only builds after the level stays high for some seconds (or it stays on, with `organ_follow` at 0). With `organ_chord` it plays several pipes, each one as detuned ranks spread in stereo. `organ_organic` makes the pipes less clean and crossfades them into blown tubes.
8. A `tanh` stage saturates the mix (`drive`).
9. A stereo reverb (a 16-line feedback delay network) adds a stone room. With `reverb_ir`, a real space replaces it.
10. A subsonic high-pass filter and a limiter protect the speakers.

All knobs, with their defaults and a short comment, are in [`wind_organ/params.py`](wind_organ/params.py).

## License

The code is MIT. The impulse responses in `wind_organ/irs/` keep their own licenses, see [`wind_organ/irs/README.md`](wind_organ/irs/README.md).
