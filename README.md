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
| `w` | Save the knobs to the `--preset` file |
| `q` | Quit |

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
5. The bourdon adds narrow whistles at the pitch, the fifth and the third. It sounds like air across a bottle.
6. The organ adds a pipe drone: sine partials at the harmonics of a deep note, plus noise that rings at the same harmonics (the air in the tube). The same gusts swell, brighten and sharpen it. It only builds after the level stays high for some seconds.
7. A `tanh` stage saturates the mix (`drive`).
8. A stereo reverb (a 16-line feedback delay network) adds a stone room.
9. A subsonic high-pass filter and a limiter protect the speakers.

All knobs, with their defaults and a short comment, are in [`wind_organ/params.py`](wind_organ/params.py).

## License

MIT
