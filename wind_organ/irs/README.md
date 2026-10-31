# Impulse responses

The files in this folder are impulse responses for `reverb_ir` (see the main README). They are not under the MIT license of the code. Each one keeps the license of the recording it comes from.

## es_culleram.wav

The Cave of Es Culleram (Sant Joan de Labritja, Ibiza): a Punic sanctuary cave in limestone. The sound dies in about 1 s (T30 of 0.7 to 0.8 s at 500 Hz to 1 kHz). The walls are close, so the first echoes come very early.

- Source: Moreno Iglesias, Diego; Torres Gomariz, Octavio; Alvarez Morales, Lidia; Díaz-Andreu, Margarita. "Acoustic measurements in the Cave of Es Culleram (Ibiza)". Universitat de Barcelona, Artsoundscapes project. Zenodo, 2026. https://doi.org/10.5281/zenodo.22272277
- Recording: interior, take `EC_T2_S01_R03_ZM_take1_WY_Raw.wav` (source S01, receiver R03), from the Zylia ZM-1 microphone, measured in October 2023.
- License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). You can use it, change it and share it, but not for commercial use. Credit the authors, and share your changed versions under the same license.
- Changes made here: the W and Y channels (mid and side) are turned into left and right (L = W + Y, R = W - Y), and the peak is scaled to 0.9. The engine also cuts the start and the direct sound, and sets the level when it reads the file.
