# wind organ

A real-time synth that makes wind, and a pipe organ that the wind blows through.

This is the sound engine of the [theremin of winds](https://github.com/pierreleripoll/theremin-of-winds), without the theremin. It has no input device. You give it two numbers per audio block, a pitch and a level, and it gives you stereo audio. The sound is built from scratch: no samples, no recordings.

## Install

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[play]'
```

The `play` extra adds `sounddevice`, which the command line player needs. The engine alone needs only `numpy` and `scipy`.

## Play

```bash
.venv/bin/python -m wind_organ                      # keyboard UI
.venv/bin/python -m wind_organ --organ              # start with the organ on
.venv/bin/python -m wind_organ --hold 38            # organ holds one note (MIDI 38 = D2)
.venv/bin/python -m wind_organ --no-tui             # plays by itself, Ctrl+C to stop
.venv/bin/python -m wind_organ --render out.wav --seconds 30 --seed 1
.venv/bin/python -m wind_organ --preset preset.json # load knob values from a JSON file
.venv/bin/python -m wind_organ --preset presets/chords_in_wind.json --chords piece --floor 0.4
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
4. A slow random LFO (low-frequency oscillator, a slow control signal) swells the level. These are the gusts. With `gust_layers` above 1, faster gust layers add turbulence.
5. The bourdon adds narrow whistles at the pitch, the fifth and the third. It sounds like air across a bottle. The wind chord does the same at the notes of `organ_chord`.
6. The organ adds a pipe drone: sine partials at the harmonics of a deep note, plus noise that rings at the same harmonics (the air in the tube). The same gusts swell, brighten and sharpen it. It only builds after the level stays high for some seconds (or it stays on, with `organ_follow` at 0). With `organ_chord` it plays several pipes, each one as detuned ranks spread in stereo.
7. A `tanh` stage saturates the mix (`drive`).
8. A stereo reverb (a 16-line feedback delay network) adds a stone room.
9. A subsonic high-pass filter and a limiter protect the speakers.

All knobs, with their defaults and a short comment, are in [`wind_organ/params.py`](wind_organ/params.py).

## License

MIT
