# Live Spectrogram Player

Live Spectrogram Player is a Windows desktop audio player with a polished dark CustomTkinter interface, a custom spectrogram app icon, and a Spek-style live scrolling spectrogram. It can draw from the player samples, a live input, a speaker loopback device, or a physical 2.1 hardware return input.

## Public build

Build a single-file Windows executable:

```powershell
.\build_exe.ps1
```

Build a folder-based release instead:

```powershell
.\build_exe.ps1 -OneDir
```

The executable is written to `dist\LiveSpectrogramPlayer.exe` for the single-file build, or `dist\LiveSpectrogramPlayer\LiveSpectrogramPlayer.exe` for the folder build.

## Notes

- Start with `Speaker loopback` for post-processing checks. Many drivers expose the processed speaker stream there, and it avoids extra cabling.
- `Sound card input` and `Speaker loopback` can show different spectra because they can represent different points in the playback chain.
- If a sound card does not include its private hardware DSP effects in speaker loopback, route the physical processed outputs back into a line-level audio interface and select that interface as `Hardware 2.1 return`.
- `Hardware 2.1 return` expects a 3+ input interface. Map the front-left, front-right, and sub/LFE cables in the `2.1 return` row, then use `2.1 L+R+Sub` to draw the combined physical output.
- Use `Live rate` to leave live capture on the device default with `Auto`, or type/select a custom live capture rate such as `8 kHz`, `22.05 kHz`, `96 kHz`, `192000`, or `384 kHz`.
- The spectrogram uses a Spek-like dark heatmap, a Nyquist default view, and a -120 dBFS to 0 dBFS colour legend with display calibration matched against Spek.
- Audio files are checked for supported extension, file size, channel count, sample rate, and estimated decoded memory before loading to avoid unsafe or accidental huge decodes.
- Audio loading validates and reads from the same open file handle, keeps only the playback channels in memory, and reads files in bounded blocks.
- Spectrogram rendering runs on a background worker with a smoothed live cadence to keep the interface responsive during live loopback capture.
- Spectrogram rendering and redraws pause during native window move/resize operations so dragging the app stays smooth.
- Live loopback/input capture drops visual spectrogram chunks during window dragging so audio capture does not fight the Windows move loop.
- Speaker loopback uses larger capture blocks than direct live input to reduce background recorder wakeups while preserving the live spectrogram.
- Speaker loopback reads use larger capture chunks and suppress repeated `soundcard` discontinuity warnings that can appear when running from a console.
- Use `Reverse live` when a live loopback or hardware return spectrogram-art image appears as a negative compared with the file view.
- Very high sample rate files are automatically resampled for playback to the selected output device rate using SOXR.
- Large files are loaded and prepared in background threads so the interface remains responsive.

## Dependencies for source builds

```powershell
python -m pip install -r requirements.txt pyinstaller
```
