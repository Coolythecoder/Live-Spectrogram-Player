#!/usr/bin/env python3
"""
Live spectrogram audio player.

The application is intentionally kept in one file so public builds are easy to
audit. The main flow is:

1. CustomTkinter builds the controls and embeds a Matplotlib spectrogram canvas.
2. soundfile loads local audio after size/type/metadata safety checks.
3. sounddevice plays the selected file and can capture a direct input stream.
4. soundcard captures speaker loopback when the user wants post-processing.
5. A small ring buffer feeds a background FFT worker so the UI stays responsive.

Dependencies:
    python -m pip install numpy matplotlib sounddevice soundfile

Run:
    python live_spectrogram_player.py
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
import ctypes
import warnings
from importlib import import_module
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import numpy as np
import matplotlib
import customtkinter as ctk
import sounddevice as sd
import soundfile as sf

matplotlib.use("TkAgg")
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

APP_NAME = "Live Spectrogram Player"
APP_VERSION = "1.0.0"
RETURN_MIX_MODE = "2.1 L+R+Sub"
RETURN_SOURCE_LABEL = "Hardware 2.1 return"
APP_ICON_PATH = ("assets", "app_icon.ico")

# UI palette shared by CustomTkinter controls and the Matplotlib frame.
APP_BG = "#151820"
PANEL_BG = "#202633"
PANEL_SUBTLE_BG = "#2a3140"
PANEL_BORDER = "#394354"
CONTROL_BG = "#171c25"
CONTROL_HOVER_BG = "#333c4b"
ACCENT = "#2f8bd8"
ACCENT_HOVER = "#2576bb"
TEXT_PRIMARY = "#eef2f7"
TEXT_MUTED = "#aeb8c7"
GRAPH_BG = "#0c0d10"

# Live capture is intentionally chunked differently for direct input and
# speaker loopback. Loopback wakes less often, which helps window dragging and
# general UI responsiveness while preserving the moving spectrogram.
LIVE_INPUT_BLOCK_SECONDS = 0.04
LOOPBACK_BLOCK_SECONDS = 0.10
WINDOW_INTERACTION_PAUSE_SECONDS = 0.18
MAX_VISUAL_CHUNKS_PER_UI_TICK = 24
SPEK_DISPLAY_GAIN_DB = -7.0

# File-loading guardrails. They reject accidental huge files and malformed
# metadata before the decoder allocates the playback buffer.
SUPPORTED_AUDIO_EXTENSIONS = frozenset((".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"))
MAX_AUDIO_FILE_BYTES = 3 * 1024 * 1024 * 1024
MAX_DECODED_AUDIO_BYTES = 2 * 1024 * 1024 * 1024
MAX_AUDIO_CHANNELS = 32
MAX_AUDIO_SAMPLE_RATE = 768_000
MAX_PLAYBACK_CHANNELS = 2
AUDIO_LOAD_BLOCK_FRAMES = 65_536
LIVE_SAMPLE_RATE_OPTIONS = (
    "Auto",
    "8 kHz",
    "11.025 kHz",
    "12 kHz",
    "16 kHz",
    "22.05 kHz",
    "24 kHz",
    "32 kHz",
    "44.1 kHz",
    "48 kHz",
    "88.2 kHz",
    "96 kHz",
    "176.4 kHz",
    "192 kHz",
    "384 kHz",
)
COLORBAR_DB_TICKS = (-120, -100, -80, -60, -40, -20, 0)
COLORBAR_DB_LABELS = tuple(f"{tick} dB" for tick in COLORBAR_DB_TICKS)

# Hand-tuned colour stops that mimic Spek's classic dark heatmap.
SPEK_COLOR_STOPS = (
    (0.000000, "#000007"),  # -120 dB
    (0.083333, "#00003c"),  # -110 dB
    (0.166667, "#1f0066"),  # -100 dB
    (0.250000, "#53007c"),  #  -90 dB
    (0.333333, "#85007d"),  #  -80 dB
    (0.416667, "#b10066"),  #  -70 dB
    (0.500000, "#d3003e"),  #  -60 dB
    (0.583333, "#ed0009"),  #  -50 dB
    (0.666667, "#fc5800"),  #  -40 dB
    (0.750000, "#ffb300"),  #  -30 dB
    (0.833333, "#ffed41"),  #  -20 dB
    (0.916667, "#ffff9e"),  #  -10 dB
    (1.000000, "#fffffe"),  #    0 dB
)

try:
    import soundcard as sc
except ImportError:  # Keep the player usable without optional loopback support.
    sc = None

warnings.filterwarnings(
    "ignore",
    message=r"data discontinuity in recording",
    category=Warning,
    module=r"soundcard\.mediafoundation",
)


def app_resource_path(*parts: str) -> Path:
    """Return a bundled asset path for source and PyInstaller builds."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root.joinpath(*parts)


def format_time(seconds: float) -> str:
    """Return seconds as M:SS or H:MM:SS."""
    seconds = max(0.0, float(seconds))
    whole = int(seconds + 0.5)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_bytes(value: int) -> str:
    """Return a compact binary size label."""
    size = float(max(0, int(value)))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


class LiveSpectrogramPlayer:
    """Owns the Tk UI, audio streams, capture workers, and spectrogram state.

    Tkinter widgets are only touched from the GUI thread. Audio callbacks and
    background workers communicate back through queues, which `_update_gui()`
    drains on a short timer.
    """

    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.root.title(f"{APP_NAME} {APP_VERSION}")
        self.root.geometry("1120x760")
        self.root.minsize(980, 660)
        self.root.configure(fg_color=APP_BG)
        self._set_window_icon()

        # Audio state. audio is always shaped (frames, channels).
        self.audio: np.ndarray | None = None
        self.mono: np.ndarray | None = None
        self.samplerate: int | None = None
        self.path: Path | None = None
        self.frame_index = 0
        self.state_lock = threading.Lock()
        self.default_music_dir = self._default_music_folder()

        # Playback stream state.
        self.playback_audio: np.ndarray | None = None
        self.playback_samplerate: int | None = None
        self.playback_source_key: tuple[str | None, int, int] | None = None
        self.stream: sd.OutputStream | None = None
        self.playing = False
        self.end_reached = False
        self.stream_finished = False
        self.loop_enabled = False
        self.volume = 1.0
        self.output_device_map: dict[str, int | None] = {}
        self.loading = False
        self.load_token = 0
        self.load_result_queue: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self.resampling = False
        self.resample_token = 0
        self.pending_play_after_resample = False
        self.resample_result_queue: queue.Queue[tuple[Any, ...]] = queue.Queue()

        # Spectrogram/visualisation settings.
        self.window_seconds = 8.0
        self.display_max_hz = 96_000.0
        self.n_fft = 8192
        self.hop_length = 512
        self.db_floor = -120.0
        self.ui_interval_ms = 33
        self.spectrogram_update_interval_seconds = 0.08
        self.max_spectrogram_columns = 480
        self.spek_cmap = self._create_spek_colormap()

        self.visual_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=120)
        self.status_queue: queue.Queue[str] = queue.Queue(maxsize=8)
        self.spectrogram_result_queue: queue.Queue[tuple[Any, ...]] = queue.Queue(maxsize=2)
        self.ring: np.ndarray | None = None
        self.ring_index = 0
        self.spectrogram_dirty = True
        self.spectrogram_rendering = False
        self.spectrogram_render_token = 0
        self.spectrogram_last_request = 0.0
        self.spectrogram_source = "loopback"
        self.ring_channels = 1
        self.window_interaction_until = 0.0
        self.native_window_move_active = False
        self.root_geometry_state: tuple[int, int, int, int] | None = None
        self.canvas_draw_pending = False
        self.native_hwnd: int | None = None
        self.native_wndproc: Any | None = None
        self.native_previous_wndproc: int | None = None
        self.native_call_window_proc: Any | None = None
        self.native_set_window_long: Any | None = None

        # Live input capture state for post-effects / loopback spectrograms.
        self.capture_stream: sd.InputStream | None = None
        self.capture_active = False
        self.capture_samplerate: int | None = None
        self.capture_frame_count = 0
        self.capture_channels = 0
        self.capture_peak_db = self.db_floor
        self.capture_rms_db = self.db_floor
        self.return_channel_indices = (0, 1, 2)
        self.input_device_map: dict[str, int | None] = {}
        self.loopback_active = False
        self.loopback_samplerate: int | None = None
        self.loopback_frame_count = 0
        self.loopback_channels = 0
        self.loopback_peak_db = self.db_floor
        self.loopback_rms_db = self.db_floor
        self.loopback_thread: threading.Thread | None = None
        self.loopback_stop_event = threading.Event()
        self.loopback_device_map: dict[str, Any] = {}
        self.player_peak_db = self.db_floor
        self.player_rms_db = self.db_floor

        # UI variables.
        self.position_var = tk.DoubleVar(value=0.0)
        self.volume_var = tk.DoubleVar(value=100.0)
        self.loop_var = tk.BooleanVar(value=False)
        self.invert_live_var = tk.BooleanVar(value=False)
        self.spectrogram_source_var = tk.StringVar(value="Speaker loopback")
        self.input_device_var = tk.StringVar(value="System default")
        self.output_device_var = tk.StringVar(value="System default")
        self.loopback_device_var = tk.StringVar(value="Default speaker")
        self.live_samplerate_var = tk.StringVar(value="Auto")
        self.live_samplerate_last_value = self.live_samplerate_var.get()
        self.channel_mode_var = tk.StringVar(value="Front L+R")
        self.channel_mode = self.channel_mode_var.get()
        self.return_left_var = tk.StringVar(value="Input 1")
        self.return_right_var = tk.StringVar(value="Input 2")
        self.return_sub_var = tk.StringVar(value="Input 3")
        self.window_var = tk.StringVar(value="8 s")
        self.frequency_var = tk.StringVar(value="Nyquist")
        self.time_var = tk.StringVar(value="0:00 / 0:00")
        self.file_var = tk.StringVar(value="Open an audio file to start.")
        self.status_var = tk.StringVar(value="Idle")
        self.meter_var = tk.StringVar(value="Draw: speaker loopback (stopped)")
        self.dragging_slider = False

        self._build_ui()
        self._reset_spectrogram_image()
        self._bind_shortcuts()
        self.root.bind("<Configure>", self._on_root_configure, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after_idle(self._install_native_window_move_hook)
        self.root.after(self.ui_interval_ms, self._update_gui)

    # ---------- UI ----------

    def _set_window_icon(self) -> None:
        """Apply the bundled icon when running from source or PyInstaller."""
        icon_path = app_resource_path(*APP_ICON_PATH)
        if not icon_path.exists():
            return

        try:
            self.root.iconbitmap(default=str(icon_path))
        except tk.TclError:
            pass

    def _build_ui(self) -> None:
        """Create the controls and embedded Matplotlib spectrogram canvas."""
        heading_font = ctk.CTkFont(size=15, weight="bold")
        label_font = ctk.CTkFont(size=12)
        small_font = ctk.CTkFont(size=11)
        row_style = {
            "corner_radius": 8,
            "fg_color": PANEL_BG,
            "border_width": 1,
            "border_color": PANEL_BORDER,
        }
        button_style = {
            "height": 30,
            "corner_radius": 6,
            "fg_color": ACCENT,
            "hover_color": ACCENT_HOVER,
        }
        secondary_button_style = {
            "height": 30,
            "corner_radius": 6,
            "fg_color": PANEL_SUBTLE_BG,
            "hover_color": CONTROL_HOVER_BG,
            "border_width": 1,
            "border_color": PANEL_BORDER,
        }
        combo_style = {
            "height": 30,
            "corner_radius": 6,
            "fg_color": CONTROL_BG,
            "border_color": PANEL_BORDER,
            "button_color": PANEL_SUBTLE_BG,
            "button_hover_color": CONTROL_HOVER_BG,
            "text_color": TEXT_PRIMARY,
            "dropdown_fg_color": PANEL_BG,
            "dropdown_hover_color": CONTROL_HOVER_BG,
            "dropdown_text_color": TEXT_PRIMARY,
        }

        def row(top: int = 0, bottom: int = 7) -> ctk.CTkFrame:
            frame = ctk.CTkFrame(self.root, **row_style)
            frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(top, bottom))
            return frame

        def subtle_label(parent: ctk.CTkFrame, text: str) -> ctk.CTkLabel:
            label = ctk.CTkLabel(parent, text=text, text_color=TEXT_MUTED, font=label_font)
            label.pack(side=tk.LEFT, pady=8)
            return label

        controls = row(top=12, bottom=7)
        ctk.CTkLabel(
            controls,
            text=f"{APP_NAME} {APP_VERSION}",
            text_color=TEXT_PRIMARY,
            font=heading_font,
        ).pack(side=tk.LEFT, padx=(10, 18), pady=8)

        self.open_button = ctk.CTkButton(
            controls, text="Open...", command=self.open_file, width=88, **button_style
        )
        self.open_button.pack(side=tk.LEFT, pady=8)

        self.play_button = ctk.CTkButton(
            controls,
            text="Play",
            command=self.toggle_play,
            state=tk.DISABLED,
            width=78,
            **button_style,
        )
        self.play_button.pack(side=tk.LEFT, padx=(8, 0), pady=8)

        self.stop_button = ctk.CTkButton(
            controls,
            text="Stop",
            command=self.stop,
            state=tk.DISABLED,
            width=78,
            **secondary_button_style,
        )
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0), pady=8)

        self.loop_check = ctk.CTkCheckBox(
            controls,
            text="Loop",
            variable=self.loop_var,
            command=self._on_loop_changed,
            width=76,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_PRIMARY,
            font=label_font,
        )
        self.loop_check.pack(side=tk.LEFT, padx=(12, 0), pady=8)

        ctk.CTkLabel(
            controls,
            textvariable=self.status_var,
            fg_color=PANEL_SUBTLE_BG,
            corner_radius=6,
            text_color=TEXT_MUTED,
            font=small_font,
        ).pack(side=tk.RIGHT, padx=(12, 10), pady=8, ipadx=12, ipady=4)

        file_bar = row(bottom=7)
        file_label = ctk.CTkLabel(
            file_bar,
            textvariable=self.file_var,
            anchor="w",
            text_color=TEXT_PRIMARY,
            font=label_font,
        )
        file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=7)

        progress_frame = row(bottom=7)

        self.position_scale = ctk.CTkSlider(
            progress_frame,
            variable=self.position_var,
            from_=0.0,
            to=1.0,
            orientation="horizontal",
            command=self._on_slider_move,
            progress_color=ACCENT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            fg_color=CONTROL_BG,
        )
        self.position_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0), pady=8)
        self.position_scale.bind("<ButtonPress-1>", self._on_slider_press)
        self.position_scale.bind("<ButtonRelease-1>", self._on_slider_release)

        ctk.CTkLabel(
            progress_frame,
            textvariable=self.time_var,
            width=135,
            text_color=TEXT_PRIMARY,
            font=label_font,
        ).pack(
            side=tk.LEFT,
            padx=(10, 10),
            pady=8,
        )

        options = row(bottom=7)

        subtle_label(options, "Volume").pack_configure(padx=(10, 0))
        self.volume_scale = ctk.CTkSlider(
            options,
            variable=self.volume_var,
            from_=0.0,
            to=125.0,
            orientation="horizontal",
            command=self._on_volume_changed,
            width=140,
            progress_color=ACCENT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            fg_color=CONTROL_BG,
        )
        self.volume_scale.pack(side=tk.LEFT, padx=(8, 18), pady=8)

        subtle_label(options, "Window")
        self.window_combo = ctk.CTkComboBox(
            options,
            variable=self.window_var,
            values=("4 s", "8 s", "15 s", "30 s"),
            width=78,
            state="readonly",
            command=self._on_window_changed,
            **combo_style,
        )
        self.window_combo.pack(side=tk.LEFT, padx=(8, 18), pady=8)

        subtle_label(options, "Max freq")
        self.frequency_combo = ctk.CTkComboBox(
            options,
            variable=self.frequency_var,
            values=("4 kHz", "8 kHz", "12 kHz", "18 kHz", "20 kHz", "Nyquist"),
            width=96,
            state="readonly",
            command=self._on_frequency_changed,
            **combo_style,
        )
        self.frequency_combo.pack(side=tk.LEFT, padx=(8, 18), pady=8)

        subtle_label(options, "Channel")
        self.channel_combo = ctk.CTkComboBox(
            options,
            variable=self.channel_mode_var,
            values=(
                "Front L+R",
                RETURN_MIX_MODE,
                "Left",
                "Right",
                "Sub/LFE",
                "Center",
                "Side L-R",
                "Loudest",
                "Mix",
            ),
            width=128,
            state="readonly",
            command=self._on_channel_mode_changed,
            **combo_style,
        )
        self.channel_combo.pack(side=tk.LEFT, padx=(8, 0), pady=8)

        self.invert_live_check = ctk.CTkCheckBox(
            options,
            text="Reverse live",
            variable=self.invert_live_var,
            command=self._on_invert_live_changed,
            width=104,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_PRIMARY,
            font=label_font,
        )
        self.invert_live_check.pack(side=tk.LEFT, padx=(14, 10), pady=8)

        monitor = row(bottom=7)

        subtle_label(monitor, "Spectrogram").pack_configure(padx=(10, 0))
        self.source_combo = ctk.CTkComboBox(
            monitor,
            variable=self.spectrogram_source_var,
            values=("Player samples", "Sound card input", "Speaker loopback", RETURN_SOURCE_LABEL),
            width=190,
            state="readonly",
            command=self._on_spectrogram_source_changed,
            **combo_style,
        )
        self.source_combo.pack(side=tk.LEFT, padx=(8, 14), pady=8)

        self.capture_button = ctk.CTkButton(
            monitor,
            text="Start loopback",
            command=self.toggle_capture,
            state=tk.NORMAL,
            width=124,
            **button_style,
        )
        self.capture_button.pack(side=tk.LEFT, padx=(0, 14), pady=8)

        subtle_label(monitor, "Live rate")
        self.live_samplerate_combo = ctk.CTkComboBox(
            monitor,
            variable=self.live_samplerate_var,
            values=LIVE_SAMPLE_RATE_OPTIONS,
            width=120,
            state=tk.NORMAL,
            command=self._on_live_samplerate_changed,
            **combo_style,
        )
        self.live_samplerate_combo.pack(side=tk.LEFT, padx=(8, 14), pady=8)
        self.live_samplerate_combo.bind("<Return>", self._on_live_samplerate_changed)
        self.live_samplerate_combo.bind("<FocusOut>", self._on_live_samplerate_changed)

        self.refresh_devices_button = ctk.CTkButton(
            monitor,
            text="Refresh",
            command=self.refresh_audio_devices,
            width=84,
            **secondary_button_style,
        )
        self.refresh_devices_button.pack(side=tk.LEFT, padx=(0, 10), pady=8)

        devices = row(bottom=7)
        subtle_label(devices, "Input").pack_configure(padx=(10, 0))
        self.input_combo = ctk.CTkComboBox(
            devices,
            variable=self.input_device_var,
            width=410,
            state="readonly",
            command=self._on_input_device_changed,
            **combo_style,
        )
        self.input_combo.pack(side=tk.LEFT, padx=(8, 18), pady=8)

        subtle_label(devices, "Output")
        self.output_combo = ctk.CTkComboBox(
            devices,
            variable=self.output_device_var,
            width=410,
            state="readonly",
            command=self._on_output_device_changed,
            **combo_style,
        )
        self.output_combo.pack(side=tk.LEFT, padx=(8, 10), pady=8)

        loopback = row(bottom=7)

        subtle_label(loopback, "Loopback").pack_configure(padx=(10, 0))
        self.loopback_combo = ctk.CTkComboBox(
            loopback,
            variable=self.loopback_device_var,
            width=600,
            state="readonly",
            command=self._on_loopback_device_changed,
            **combo_style,
        )
        self.loopback_combo.pack(side=tk.LEFT, padx=(8, 10), pady=8)

        return_channels = row(bottom=7)

        subtle_label(return_channels, "2.1 return").pack_configure(padx=(10, 0))
        subtle_label(return_channels, "L").pack_configure(padx=(14, 4))
        self.return_left_combo = ctk.CTkComboBox(
            return_channels,
            variable=self.return_left_var,
            values=("Input 1", "Input 2", "Input 3"),
            width=96,
            state=tk.DISABLED,
            command=self._on_return_channel_map_changed,
            **combo_style,
        )
        self.return_left_combo.pack(side=tk.LEFT, pady=8)

        subtle_label(return_channels, "R").pack_configure(padx=(14, 4))
        self.return_right_combo = ctk.CTkComboBox(
            return_channels,
            variable=self.return_right_var,
            values=("Input 1", "Input 2", "Input 3"),
            width=96,
            state=tk.DISABLED,
            command=self._on_return_channel_map_changed,
            **combo_style,
        )
        self.return_right_combo.pack(side=tk.LEFT, pady=8)

        subtle_label(return_channels, "Sub").pack_configure(padx=(14, 4))
        self.return_sub_combo = ctk.CTkComboBox(
            return_channels,
            variable=self.return_sub_var,
            values=("Input 1", "Input 2", "Input 3"),
            width=96,
            state=tk.DISABLED,
            command=self._on_return_channel_map_changed,
            **combo_style,
        )
        self.return_sub_combo.pack(side=tk.LEFT, pady=8)

        self.refresh_audio_devices()

        meter = ctk.CTkFrame(
            self.root,
            corner_radius=8,
            fg_color=PANEL_SUBTLE_BG,
            border_width=1,
            border_color=PANEL_BORDER,
        )
        meter.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(0, 8))
        ctk.CTkLabel(
            meter,
            textvariable=self.meter_var,
            anchor="w",
            text_color=TEXT_MUTED,
            font=small_font,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=7)

        self.fig = Figure(figsize=(9, 5), dpi=100, facecolor=GRAPH_BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Spectrogram - speaker loopback")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (kHz)")
        self._style_spek_axes()

        initial = np.full((128, 128), self.db_floor, dtype=np.float32)
        self.image = self.ax.imshow(
            initial,
            origin="lower",
            aspect="auto",
            interpolation="bilinear",
            extent=(0.0, self.window_seconds, 0.0, self.display_max_hz / 1000.0),
            vmin=self.db_floor,
            vmax=0.0,
            cmap=self.spek_cmap,
        )
        self.colorbar = self.fig.colorbar(self.image, ax=self.ax)
        self._style_spek_colorbar()
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(
            bg=GRAPH_BG,
            highlightthickness=1,
            highlightbackground=PANEL_BORDER,
            highlightcolor=PANEL_BORDER,
        )
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _event: self.open_file())
        self.root.bind("<space>", self._on_space_pressed)
        self.root.bind("<Left>", lambda _event: self.seek_relative(-5.0))
        self.root.bind("<Right>", lambda _event: self.seek_relative(5.0))

    def _on_root_configure(self, event: tk.Event[Any]) -> None:
        if event.widget is not self.root:
            return

        geometry_state = (
            int(getattr(event, "width", 0)),
            int(getattr(event, "height", 0)),
            int(getattr(event, "x", 0)),
            int(getattr(event, "y", 0)),
        )
        if geometry_state == self.root_geometry_state:
            return

        self.root_geometry_state = geometry_state
        self.window_interaction_until = time.monotonic() + WINDOW_INTERACTION_PAUSE_SECONDS

    def _window_interaction_active(self) -> bool:
        return self.native_window_move_active or time.monotonic() < self.window_interaction_until

    def _request_canvas_draw(self) -> None:
        if self._window_interaction_active():
            self.canvas_draw_pending = True
            return

        self.canvas_draw_pending = False
        self.canvas.draw_idle()

    def _flush_deferred_canvas_draw(self) -> None:
        if self.canvas_draw_pending and not self._window_interaction_active():
            self.canvas_draw_pending = False
            self.canvas.draw_idle()

    @staticmethod
    def _discard_queue_items(target_queue: queue.Queue[Any]) -> None:
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                return

    def _mark_window_move_active(self) -> None:
        self.native_window_move_active = True
        self.window_interaction_until = time.monotonic() + WINDOW_INTERACTION_PAUSE_SECONDS
        self.canvas_draw_pending = True
        self.spectrogram_render_token += 1
        self.spectrogram_rendering = False
        self.spectrogram_dirty = True
        self._discard_queue_items(self.visual_queue)
        self._discard_queue_items(self.spectrogram_result_queue)

    def _mark_window_move_finished(self) -> None:
        self.native_window_move_active = False
        self.window_interaction_until = time.monotonic() + WINDOW_INTERACTION_PAUSE_SECONDS
        self.canvas_draw_pending = True
        self.spectrogram_dirty = True

    def _install_native_window_move_hook(self) -> None:
        """Pause expensive redraw work while Windows is moving/resizing us."""
        if sys.platform != "win32" or self.native_wndproc is not None:
            return

        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            client_hwnd = int(self.root.winfo_id())
            get_ancestor = user32.GetAncestor
            get_ancestor.argtypes = (ctypes.c_void_p, ctypes.c_uint)
            get_ancestor.restype = ctypes.c_void_p
            hwnd = int(get_ancestor(ctypes.c_void_p(client_hwnd), 2) or client_hwnd)
            call_window_proc = user32.CallWindowProcW

            pointer_sized = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
            if ctypes.sizeof(ctypes.c_void_p) == 8:
                set_window_long = user32.SetWindowLongPtrW
            else:
                set_window_long = user32.SetWindowLongW

            wndproc_type = ctypes.WINFUNCTYPE(
                pointer_sized,
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_size_t,
                pointer_sized,
            )

            call_window_proc.argtypes = (
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_size_t,
                pointer_sized,
            )
            call_window_proc.restype = pointer_sized
            set_window_long.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
            set_window_long.restype = ctypes.c_void_p

            def window_proc(hwnd_arg: int, message: int, wparam: int, lparam: int) -> int:
                wm_entersizemove = 0x0231
                wm_exitsizemove = 0x0232
                wm_sizing = 0x0214
                wm_moving = 0x0216

                try:
                    if message == wm_entersizemove:
                        self._mark_window_move_active()
                    elif message in {wm_sizing, wm_moving}:
                        self.window_interaction_until = time.monotonic() + WINDOW_INTERACTION_PAUSE_SECONDS
                        self.canvas_draw_pending = True
                    elif message == wm_exitsizemove:
                        self._mark_window_move_finished()
                except Exception:
                    pass

                previous = self.native_previous_wndproc
                if previous is None:
                    return 0
                return int(
                    call_window_proc(
                        ctypes.c_void_p(previous),
                        ctypes.c_void_p(hwnd_arg),
                        message,
                        wparam,
                        lparam,
                    )
                )

            native_wndproc = wndproc_type(window_proc)
            previous = set_window_long(
                ctypes.c_void_p(hwnd),
                -4,
                ctypes.cast(native_wndproc, ctypes.c_void_p),
            )
            if not previous:
                error = ctypes.get_last_error()
                if error:
                    raise ctypes.WinError(error)

            self.native_hwnd = hwnd
            self.native_wndproc = native_wndproc
            self.native_previous_wndproc = int(previous)
            self.native_call_window_proc = call_window_proc
            self.native_set_window_long = set_window_long
        except Exception as exc:  # noqa: BLE001 - hook is an optimization, not core app behavior.
            try:
                self.status_queue.put_nowait(f"Window move hook unavailable: {exc}")
            except queue.Full:
                pass

    def _restore_native_window_move_hook(self) -> None:
        if (
            sys.platform != "win32"
            or self.native_hwnd is None
            or self.native_previous_wndproc is None
            or self.native_set_window_long is None
        ):
            return

        try:
            self.native_set_window_long(
                ctypes.c_void_p(self.native_hwnd),
                -4,
                ctypes.c_void_p(self.native_previous_wndproc),
            )
        except Exception:
            pass
        finally:
            self.native_hwnd = None
            self.native_previous_wndproc = None
            self.native_wndproc = None
            self.native_call_window_proc = None
            self.native_set_window_long = None

    def _on_space_pressed(self, _event: tk.Event[Any]) -> str:
        if self.audio is not None:
            self.toggle_play()
        return "break"

    def _on_loop_changed(self) -> None:
        self.loop_enabled = bool(self.loop_var.get())

    def _on_volume_changed(self, value: str) -> None:
        self.volume = max(0.0, min(float(value) / 100.0, 1.25))

    def _on_window_changed(self, _event: tk.Event[Any] | None = None) -> None:
        value = self.window_var.get().split()[0]
        self.window_seconds = float(value)
        self._reset_visual_buffer()
        self._reset_spectrogram_image()
        self._request_canvas_draw()

    def _on_frequency_changed(self, _event: tk.Event[Any] | None = None) -> None:
        value = self.frequency_var.get()
        if value == "Nyquist":
            self.display_max_hz = 96_000.0
        else:
            self.display_max_hz = float(value.split()[0]) * 1000.0
        self._reset_spectrogram_image()
        self._request_canvas_draw()

    def _on_channel_mode_changed(self, _event: tk.Event[Any] | None = None) -> None:
        self.channel_mode = self.channel_mode_var.get()
        self._reset_visual_buffer()
        self._reset_spectrogram_image()
        self._update_meter_text()
        self._request_canvas_draw()

    def _on_invert_live_changed(self) -> None:
        self.spectrogram_render_token += 1
        self.spectrogram_dirty = True
        self._update_meter_text()
        self._start_spectrogram_render()

    def refresh_audio_devices(self) -> None:
        """Refresh device labels and the lookup maps used by combo boxes."""
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as exc:  # noqa: BLE001 - user-facing status
            self.status_var.set(f"Device refresh failed: {exc}")
            return

        previous_input = self.input_device_var.get()
        previous_output = self.output_device_var.get()
        self.input_device_map = {"System default": None}
        self.output_device_map = {"System default": None}

        for index, device in enumerate(devices):
            hostapi_name = hostapis[device["hostapi"]]["name"]
            name = self._clean_device_name(device["name"])
            label = f"{index}: {name} [{hostapi_name}]"
            if int(device["max_input_channels"]) > 0:
                self.input_device_map[label] = index
            if int(device["max_output_channels"]) > 0:
                self.output_device_map[label] = index

        input_values = tuple(self.input_device_map)
        output_values = tuple(self.output_device_map)
        self.input_combo.configure(values=list(input_values))
        self.output_combo.configure(values=list(output_values))

        if self._should_preserve_input_label(previous_input):
            self.input_device_var.set(previous_input)
        else:
            self.input_device_var.set(self._preferred_input_label_for_current_source(input_values))

        if self._should_preserve_output_label(previous_output):
            self.output_device_var.set(previous_output)
        else:
            self.output_device_var.set(self._preferred_output_label(output_values))

        self._sync_return_channel_controls()
        self._refresh_loopback_devices()
        self.status_var.set("Audio devices refreshed")

    @staticmethod
    def _clean_device_name(name: Any) -> str:
        return " ".join(str(name).split())

    @staticmethod
    def _default_music_folder() -> Path:
        music_folder = Path.home() / "Music"
        if music_folder.exists():
            return music_folder
        return Path.cwd()

    @staticmethod
    def _api_rank(label: str) -> int:
        lower = label.lower()
        if "wasapi" in lower:
            return 0
        if "directsound" in lower:
            return 1
        if "mme" in lower:
            return 2
        return 3

    @staticmethod
    def _label_contains_any(label: str, terms: tuple[str, ...]) -> bool:
        lower = label.lower()
        return any(term in lower for term in terms)

    def _preferred_input_label_for_current_source(self, labels: tuple[str, ...]) -> str:
        if self.spectrogram_source == "return":
            return self._preferred_return_input_label(labels)
        return self._preferred_input_label(labels)

    def _should_preserve_input_label(self, label: str) -> bool:
        if label not in self.input_device_map or label == "System default":
            return False

        lower = label.lower()
        if "wdm-ks" in lower:
            return False
        return True

    def _should_preserve_output_label(self, label: str) -> bool:
        if label not in self.output_device_map or label == "System default":
            return False

        lower = label.lower()
        if "wdm-ks" in lower:
            return False
        return True

    @staticmethod
    def _preferred_output_label(labels: tuple[str, ...]) -> str:
        candidates: list[tuple[int, int, str]] = []
        for index, label in enumerate(labels):
            lower = label.lower()
            if label == "System default" or "wdm-ks" in lower:
                continue
            score = LiveSpectrogramPlayer._api_rank(label) * 10
            if LiveSpectrogramPlayer._label_contains_any(label, ("speakers", "headphones")):
                score -= 4
            if "sound blaster" in lower:
                score -= 1
            candidates.append((score, index, label))

        if candidates:
            candidates.sort()
            return candidates[0][2]
        return "System default"

    @staticmethod
    def _preferred_input_label(labels: tuple[str, ...]) -> str:
        preferred_terms = ("stereo mix", "what u hear", "loopback", "monitor", "line in")
        candidates: list[tuple[int, int, str]] = []
        for index, label in enumerate(labels):
            lower = label.lower()
            if label == "System default" or "wdm-ks" in lower:
                continue
            if not LiveSpectrogramPlayer._label_contains_any(label, preferred_terms):
                continue
            candidates.append((LiveSpectrogramPlayer._api_rank(label), index, label))

        if candidates:
            candidates.sort()
            return candidates[0][2]
        return "System default"

    def _input_channel_count_for_label(self, label: str) -> int:
        if label not in self.input_device_map or label == "System default":
            return 0
        try:
            device = sd.query_devices(self.input_device_map[label], "input")
        except Exception:
            return 0
        return max(0, int(device["max_input_channels"]))

    def _preferred_return_input_label(self, labels: tuple[str, ...]) -> str:
        interface_terms = (
            "umc404",
            "u-phoria",
            "behringer",
            "interface",
            "usb audio",
            "focusrite",
            "scarlett",
            "steinberg",
            "presonus",
            "audient",
            "motu",
            "tascam",
            "komplete",
            "volt",
            "m-track",
        )
        line_terms = ("line in", "aux")
        software_terms = ("stereo mix", "what u hear", "loopback", "monitor")
        microphone_terms = ("microphone", "steam streaming", "webcam", "camera", "headset")
        virtual_terms = ("virtual", "vb-audio", "cable output", "voicemeeter")
        candidates: list[tuple[int, int, int, str]] = []
        for index, label in enumerate(labels):
            lower = label.lower()
            if label == "System default" or "wdm-ks" in lower:
                continue
            is_interface = LiveSpectrogramPlayer._label_contains_any(label, interface_terms)
            is_line = LiveSpectrogramPlayer._label_contains_any(label, line_terms)
            if not is_interface and not is_line:
                continue
            if LiveSpectrogramPlayer._label_contains_any(label, microphone_terms) and not is_interface:
                continue
            if LiveSpectrogramPlayer._label_contains_any(label, software_terms):
                continue
            if LiveSpectrogramPlayer._label_contains_any(label, virtual_terms) and not is_interface:
                continue
            channels = self._input_channel_count_for_label(label)
            if channels <= 0:
                continue

            score = LiveSpectrogramPlayer._api_rank(label) * 4
            if channels >= 3:
                score -= 20
                score += abs(channels - 4)
            else:
                score += 16
            if is_interface:
                score -= 8
            if is_line:
                score -= 3
            candidates.append((score, index, -channels, label))

        if candidates:
            candidates.sort()
            return candidates[0][3]
        return "System default"

    def _refresh_loopback_devices(self) -> None:
        previous_loopback = self.loopback_device_var.get()
        self.loopback_device_map = {}

        if sc is None:
            unavailable = "soundcard module not installed"
            self.loopback_device_map[unavailable] = None
            self.loopback_combo.configure(values=[unavailable], state=tk.DISABLED)
            self.loopback_device_var.set(unavailable)
            return

        try:
            loopbacks = [
                microphone
                for microphone in sc.all_microphones(include_loopback=True)
                if bool(getattr(microphone, "isloopback", False))
            ]
        except Exception as exc:  # noqa: BLE001 - user-facing status
            unavailable = f"Loopback refresh failed: {exc}"
            self.loopback_device_map[unavailable] = None
            self.loopback_combo.configure(values=[unavailable], state=tk.DISABLED)
            self.loopback_device_var.set(unavailable)
            return

        for index, loopback in enumerate(loopbacks):
            name = self._clean_device_name(getattr(loopback, "name", loopback))
            label = f"{index}: {name} [soundcard loopback]"
            self.loopback_device_map[label] = loopback

        if not self.loopback_device_map:
            unavailable = "No soundcard loopback devices"
            self.loopback_device_map[unavailable] = None
            self.loopback_combo.configure(values=[unavailable], state=tk.DISABLED)
            self.loopback_device_var.set(unavailable)
            return

        values = tuple(self.loopback_device_map)
        self.loopback_combo.configure(values=list(values), state="readonly")
        if previous_loopback in self.loopback_device_map:
            self.loopback_device_var.set(previous_loopback)
        else:
            self.loopback_device_var.set(self._preferred_loopback_label(values))

    @staticmethod
    def _preferred_loopback_label(labels: tuple[str, ...]) -> str:
        candidates: list[tuple[int, int, str]] = []
        for index, label in enumerate(labels):
            score = 10
            if LiveSpectrogramPlayer._label_contains_any(label, ("speakers", "headphones")):
                score -= 4
            if "sound blaster" in label.lower():
                score -= 1
            candidates.append((score, index, label))
        candidates.sort()
        return candidates[0][2]

    @staticmethod
    def _return_channel_label(index: int) -> str:
        return f"Input {index + 1}"

    @staticmethod
    def _return_channel_index(label: str) -> int:
        parts = str(label).split()
        if len(parts) >= 2 and parts[-1].isdigit():
            return max(0, int(parts[-1]) - 1)
        return 0

    def _selected_input_channel_count(self) -> int:
        try:
            device = sd.query_devices(self._selected_input_device(), "input")
        except Exception:
            return 0
        return max(0, int(device["max_input_channels"]))

    def _sync_return_channel_controls(self) -> None:
        if not hasattr(self, "return_left_combo"):
            return

        max_channels = self._selected_input_channel_count()
        visible_channels = max(3, min(max_channels or 3, 8))
        values = [self._return_channel_label(index) for index in range(visible_channels)]
        state = "readonly" if self.spectrogram_source == "return" else tk.DISABLED
        for combo in (self.return_left_combo, self.return_right_combo, self.return_sub_combo):
            combo.configure(values=values, state=state)

        defaults = (
            (self.return_left_var, "Input 1"),
            (self.return_right_var, "Input 2"),
            (self.return_sub_var, "Input 3"),
        )
        for variable, default in defaults:
            if variable.get() not in values:
                variable.set(default if default in values else values[0])

        self._update_return_channel_indices()

    def _update_return_channel_indices(self) -> None:
        self.return_channel_indices = (
            self._return_channel_index(self.return_left_var.get()),
            self._return_channel_index(self.return_right_var.get()),
            self._return_channel_index(self.return_sub_var.get()),
        )

    def _return_channel_summary(self) -> str:
        left, right, sub = self.return_channel_indices
        return (
            f"L {self._return_channel_label(left)}, "
            f"R {self._return_channel_label(right)}, "
            f"Sub {self._return_channel_label(sub)}"
        )

    def _on_return_channel_map_changed(self, _event: tk.Event[Any] | None = None) -> None:
        self._update_return_channel_indices()
        self._reset_visual_buffer()
        self._reset_spectrogram_image()
        self._update_meter_text()
        self._request_canvas_draw()
        if self.capture_active and self.spectrogram_source == "return":
            self.status_var.set(f"2.1 return map: {self._return_channel_summary()}")

    def _on_spectrogram_source_changed(self, _event: tk.Event[Any] | None = None) -> None:
        selected_source = self.spectrogram_source_var.get()
        if selected_source == "Sound card input":
            self.spectrogram_source = "capture"
        elif selected_source == "Speaker loopback":
            self.spectrogram_source = "loopback"
        elif selected_source in {"Hardware return", RETURN_SOURCE_LABEL}:
            self.spectrogram_source = "return"
        else:
            self.spectrogram_source = "player"

        if self.spectrogram_source in {"capture", "return"}:
            self.stop_loopback()
            if self.spectrogram_source == "return":
                self.input_device_var.set(
                    self._preferred_return_input_label(tuple(self.input_device_map))
                )
                self.channel_mode_var.set(RETURN_MIX_MODE)
                self.channel_mode = RETURN_MIX_MODE
                button_text = "Start 2.1 return"
                status_text = "Select a 3+ input interface, map L/R/Sub, then start return"
            else:
                button_text = "Start input"
                status_text = "Select a capture input, then start input"
            self._sync_return_channel_controls()
            self.capture_button.configure(text=button_text, state=tk.NORMAL)
            self._reset_visual_buffer()
            self._reset_spectrogram_image()
            self._update_plot_title()
            self._update_meter_text()
            self._request_canvas_draw()
            self.status_var.set(status_text)
        elif self.spectrogram_source == "loopback":
            self.stop_capture()
            self._sync_return_channel_controls()
            self.capture_button.configure(text="Start loopback", state=tk.NORMAL)
            self._reset_visual_buffer()
            self._reset_spectrogram_image()
            self._update_plot_title()
            self._update_meter_text()
            self._request_canvas_draw()
            self.status_var.set("Select a loopback speaker, then start loopback")
        else:
            self.stop_capture()
            self.stop_loopback()
            self._sync_return_channel_controls()
            self.capture_button.configure(state=tk.DISABLED)
            self._reset_visual_buffer()
            self._reset_spectrogram_image()
            self._update_plot_title()
            self._update_meter_text()
            self._request_canvas_draw()
            self.status_var.set("Spectrogram source: player samples")

    def _selected_input_device(self) -> int | None:
        return self.input_device_map.get(self.input_device_var.get())

    def _selected_output_device(self) -> int | None:
        return self.output_device_map.get(self.output_device_var.get())

    def _selected_loopback_device(self) -> Any | None:
        return self.loopback_device_map.get(self.loopback_device_var.get())

    def _label_for_input_device(self, device_index: int | None) -> str | None:
        for label, index in self.input_device_map.items():
            if index == device_index:
                return label
        return None

    @staticmethod
    def _return_input_block_reason(label: str) -> str | None:
        lower = label.lower()
        if label == "System default":
            return "System default can point at a microphone or software tap."
        if any(term in lower for term in ("microphone", "steam streaming", "webcam", "camera", "headset")):
            return "Microphone-style inputs are not the physical speaker return."
        if any(term in lower for term in ("what u hear", "stereo mix", "loopback")):
            return "Software loopback inputs do not guarantee the final analog post-processing."
        if any(term in lower for term in ("virtual", "vb-audio", "cable output", "voicemeeter")):
            return "Virtual cable inputs are not the physical soundcard speaker outputs."
        return None

    def _capture_fallback_labels(self, failed_device_index: int | None) -> list[str]:
        if failed_device_index is None:
            return []

        try:
            failed_device = sd.query_devices(failed_device_index, "input")
            hostapis = sd.query_hostapis()
        except Exception:
            return []

        failed_name = self._clean_device_name(failed_device["name"]).lower()
        failed_tokens = {
            token
            for token in failed_name.replace("(", " ").replace(")", " ").replace("-", " ").split()
            if len(token) >= 4 and token not in {"audio", "input", "device"}
        }
        preferred_apis = ("Windows WASAPI", "Windows DirectSound", "MME")
        source_terms = (
            "what u hear",
            "stereo mix",
            "line in",
            "aux",
            "loopback",
            "monitor",
            "umc404",
            "u-phoria",
            "behringer",
            "interface",
            "usb audio",
        )
        candidates: list[tuple[int, int, str]] = []

        for label, index in self.input_device_map.items():
            if index is None or index == failed_device_index:
                continue
            try:
                device = sd.query_devices(index, "input")
            except Exception:
                continue
            name = self._clean_device_name(device["name"]).lower()
            api_name = hostapis[device["hostapi"]]["name"]
            if api_name not in preferred_apis:
                continue
            name_tokens = {
                token
                for token in name.replace("(", " ").replace(")", " ").replace("-", " ").split()
                if len(token) >= 4 and token not in {"audio", "input", "device"}
            }
            name_match = bool(failed_tokens & name_tokens)
            source_match = self._label_contains_any(label, source_terms)
            if not name_match and not source_match:
                continue
            candidates.append((0 if name_match else 1, preferred_apis.index(api_name), label))

        candidates.sort(key=lambda item: item[0])
        return [label for _, _, label in candidates]

    def _loopback_default_samplerate(self, loopback_device: Any) -> int:
        loopback_name = self._clean_device_name(getattr(loopback_device, "name", ""))
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception:
            return 48_000

        exact_matches: list[tuple[bool, int]] = []
        partial_matches: list[tuple[bool, int]] = []
        for device in devices:
            if int(device["max_output_channels"]) <= 0:
                continue
            device_name = self._clean_device_name(device["name"])
            samplerate = int(device["default_samplerate"] or 48_000)
            is_wasapi = hostapis[device["hostapi"]]["name"] == "Windows WASAPI"
            if device_name == loopback_name:
                exact_matches.append((is_wasapi, samplerate))
            elif loopback_name and (loopback_name in device_name or device_name in loopback_name):
                partial_matches.append((is_wasapi, samplerate))

        matches = exact_matches or partial_matches
        if matches:
            matches.sort(key=lambda match: match[0], reverse=True)
            return matches[0][1]
        return 48_000

    @staticmethod
    def _parse_live_samplerate(value: str) -> int | None:
        text = str(value).strip()
        if not text or text.lower() in {"auto", "default", "device default"}:
            return None

        compact = text.lower().replace(",", "").replace("_", "").replace(" ", "")
        multiplier = 1.0
        if compact.endswith("khz"):
            compact = compact[:-3]
            multiplier = 1000.0
        elif compact.endswith("k"):
            compact = compact[:-1]
            multiplier = 1000.0
        elif compact.endswith("hz"):
            compact = compact[:-2]

        try:
            numeric = float(compact)
        except ValueError as exc:
            raise ValueError("Live sample rate must be Auto or a number like 96000 / 96 kHz.") from exc

        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("Live sample rate must be a positive number.")
        if multiplier == 1.0 and numeric < 1000.0:
            multiplier = 1000.0

        samplerate = int(round(numeric * multiplier))
        if samplerate < 8_000 or samplerate > 768_000:
            raise ValueError("Live sample rate must be between 8,000 Hz and 768,000 Hz.")
        return samplerate

    def _live_samplerate_for_default(self, default_samplerate: int) -> int:
        custom_samplerate = self._parse_live_samplerate(self.live_samplerate_var.get())
        if custom_samplerate is None:
            return max(1, int(default_samplerate))
        return custom_samplerate

    def _live_samplerate_status(self) -> str:
        custom_samplerate = self._parse_live_samplerate(self.live_samplerate_var.get())
        if custom_samplerate is None:
            return "Auto live rate"
        return f"Live rate {custom_samplerate:,} Hz"

    def _on_live_samplerate_changed(self, _event: Any | None = None) -> None:
        current_value = self.live_samplerate_var.get().strip()
        if current_value == self.live_samplerate_last_value:
            return

        try:
            status_text = self._live_samplerate_status()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return

        self.live_samplerate_last_value = current_value
        if self.capture_active and self.spectrogram_source in {"capture", "return"}:
            self.start_capture()
        elif self.loopback_active:
            self.start_loopback()
        elif self.spectrogram_source in {"capture", "return", "loopback"}:
            self.status_var.set(status_text)

    @staticmethod
    def _live_block_frames(samplerate: int) -> int:
        frames = int(round(max(1, samplerate) * LIVE_INPUT_BLOCK_SECONDS))
        return max(1024, min(8192, frames))

    @staticmethod
    def _loopback_block_frames(samplerate: int) -> int:
        frames = int(round(max(1, samplerate) * LOOPBACK_BLOCK_SECONDS))
        return max(2048, min(32768, frames))

    def _on_input_device_changed(self, _event: tk.Event[Any] | None = None) -> None:
        self._sync_return_channel_controls()
        if self.capture_active:
            self.start_capture()
        elif self.spectrogram_source in {"capture", "return"}:
            self.status_var.set("Input selected")

    def _on_output_device_changed(self, _event: tk.Event[Any] | None = None) -> None:
        was_playing = self.playing
        self._close_stream(abort=True)
        self.playing = False
        self.play_button.configure(text="Play")
        self.pending_play_after_resample = False
        if self.resampling:
            self.resampling = False
            self.resample_token += 1
        if self.audio is not None and self.samplerate is not None:
            target_rate = self._selected_output_samplerate()
            self.status_var.set(f"Output selected ({target_rate:,} Hz)")
            self._start_playback_resample(target_rate, auto_play=was_playing)
        else:
            self.status_var.set("Output selected")

    def _on_loopback_device_changed(self, _event: tk.Event[Any] | None = None) -> None:
        if self.loopback_active:
            self.start_loopback()
        elif self.spectrogram_source == "loopback":
            self.status_var.set("Loopback speaker selected")

    def _set_player_meter(self, samples: np.ndarray) -> None:
        self.player_peak_db, self.player_rms_db = self._signal_db(samples)

    def _set_capture_meter(self, samples: np.ndarray) -> None:
        self.capture_peak_db, self.capture_rms_db = self._signal_db(samples)

    def _set_loopback_meter(self, samples: np.ndarray) -> None:
        self.loopback_peak_db, self.loopback_rms_db = self._signal_db(samples)

    @staticmethod
    def _channel_or_silence(values: np.ndarray, channel: int) -> np.ndarray:
        if 0 <= channel < values.shape[1]:
            return values[:, channel].astype(np.float32, copy=False)
        return np.zeros(values.shape[0], dtype=np.float32)

    @staticmethod
    def _mix_channel_indices(values: np.ndarray, channels: tuple[int, ...]) -> np.ndarray:
        if not channels:
            return np.zeros(values.shape[0], dtype=np.float32)

        mixed = np.zeros(values.shape[0], dtype=np.float32)
        active_channels = 0
        for channel in channels:
            if 0 <= channel < values.shape[1]:
                mixed += values[:, channel]
                active_channels += 1

        if active_channels:
            mixed /= float(active_channels)
        return mixed.astype(np.float32, copy=False)

    @staticmethod
    def _select_channel_indices(values: np.ndarray, channels: tuple[int, ...]) -> np.ndarray:
        selected = [channel for channel in channels if 0 <= channel < values.shape[1]]
        if not selected:
            return np.zeros(values.shape[0], dtype=np.float32)
        if len(selected) == 1:
            return values[:, selected[0]].astype(np.float32, copy=False)
        return values[:, selected].astype(np.float32, copy=False)

    def _default_sub_channel_index(self, values: np.ndarray) -> int:
        if self.spectrogram_source == "return":
            return self.return_channel_indices[2]
        if values.shape[1] > 3:
            return 3
        return 2

    def _hardware_return_visual_channel(self, values: np.ndarray, mode: str) -> np.ndarray:
        """Apply the user-selected 2.1 return channel map before drawing."""
        left, right, sub = self.return_channel_indices
        if mode == RETURN_MIX_MODE:
            return self._select_channel_indices(values, (left, right, sub))
        if mode == "Front L+R":
            return self._select_channel_indices(values, (left, right))
        if mode == "Left":
            return self._channel_or_silence(values, left)
        if mode == "Right":
            return self._channel_or_silence(values, right)
        if mode == "Sub/LFE":
            return self._channel_or_silence(values, sub)
        return self._visual_channel_by_mode(values, mode)

    def _visual_channel_by_mode(self, values: np.ndarray, mode: str) -> np.ndarray:
        """Convert a multichannel block into the view requested by the UI."""
        if mode == RETURN_MIX_MODE:
            sub = self._default_sub_channel_index(values)
            if values.shape[1] == 1:
                return values[:, 0].astype(np.float32, copy=False)
            if values.shape[1] == 2:
                return self._select_channel_indices(values, (0, 1))
            return self._select_channel_indices(values, (0, 1, sub))
        if mode == "Front L+R":
            if values.shape[1] == 1:
                return values[:, 0].astype(np.float32, copy=False)
            return values[:, :2].astype(np.float32, copy=False)
        if mode == "Left":
            return values[:, 0].astype(np.float32, copy=False)
        if mode == "Right":
            channel = 1 if values.shape[1] > 1 else 0
            return values[:, channel].astype(np.float32, copy=False)
        if mode == "Sub/LFE":
            sub = self._default_sub_channel_index(values)
            if values.shape[1] > sub:
                return values[:, sub].astype(np.float32, copy=False)
            return np.zeros(values.shape[0], dtype=np.float32)
        if mode == "Center":
            if values.shape[1] > 2:
                return values[:, 2].astype(np.float32, copy=False)
            return np.zeros(values.shape[0], dtype=np.float32)
        if mode == "Side L-R":
            if values.shape[1] < 2:
                return np.zeros(values.shape[0], dtype=np.float32)
            return ((values[:, 0] - values[:, 1]) * 0.5).astype(np.float32, copy=False)
        if mode == "Loudest":
            channel_levels = np.sqrt(np.mean(values * values, axis=0))
            channel = int(np.argmax(channel_levels))
            return values[:, channel].astype(np.float32, copy=False)

        return values.astype(np.float32, copy=False)

    def _visual_channel(self, samples: np.ndarray) -> np.ndarray:
        """Normalize sample shape and route to the selected visual channel mode."""
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            return values.reshape(-1)
        if values.size == 0 or values.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)

        mode = self.channel_mode if hasattr(self, "channel_mode") else "Mix"
        if self.spectrogram_source == "return":
            return self._hardware_return_visual_channel(values, mode)
        return self._visual_channel_by_mode(values, mode)

    def _update_meter_text(self) -> None:
        invert_note = ", reversed" if self._should_invert_spectrogram_levels() else ""
        if self.spectrogram_source in {"capture", "return"}:
            state = "active" if self.capture_active else "stopped"
            samplerate = self.capture_samplerate or self._visual_samplerate()
            label = "2.1 hardware return" if self.spectrogram_source == "return" else "sound card input"
            channel_summary = (
                f", {self._return_channel_summary()}" if self.spectrogram_source == "return" else ""
            )
            self.meter_var.set(
                f"Draw: {label} "
                f"({state}, {samplerate:,} Hz, {self.capture_channels} ch, "
                f"{self.channel_mode_var.get()}{channel_summary}{invert_note}) | "
                f"peak {self.capture_peak_db:.1f} dBFS | "
                f"rms {self.capture_rms_db:.1f} dBFS"
            )
            return

        if self.spectrogram_source == "loopback":
            state = "active" if self.loopback_active else "stopped"
            samplerate = self.loopback_samplerate or self._visual_samplerate()
            self.meter_var.set(
                "Draw: speaker loopback "
                f"({state}, {samplerate:,} Hz, {self.loopback_channels} ch, "
                f"{self.channel_mode_var.get()}{invert_note}) | "
                f"peak {self.loopback_peak_db:.1f} dBFS | "
                f"rms {self.loopback_rms_db:.1f} dBFS"
            )
            return

        player_channels = int(self.audio.shape[1]) if self.audio is not None else 0

        self.meter_var.set(
            "Draw: player samples "
            f"({player_channels} ch, {self.channel_mode_var.get()}) | "
            f"peak {self.player_peak_db:.1f} dBFS | "
            f"rms {self.player_rms_db:.1f} dBFS"
        )

    def _update_plot_title(self) -> None:
        if self.spectrogram_source == "capture":
            source = "sound card input"
        elif self.spectrogram_source == "return":
            source = "2.1 hardware return"
        elif self.spectrogram_source == "loopback":
            source = "speaker loopback"
        else:
            source = "player samples"
        self.ax.set_title(f"Spectrogram - {source}")
        self._style_spek_axes()

    def _update_capture_silence_status(self) -> None:
        if not self.capture_active or not self.playing:
            return
        if self.capture_frame_count < (self.capture_samplerate or 1):
            return
        if self.capture_peak_db <= -80.0:
            if self.spectrogram_source == "return":
                self.status_var.set("2.1 hardware return is near silent")
            else:
                self.status_var.set("Capture input is near silent")

    def _update_loopback_silence_status(self) -> None:
        if not self.loopback_active or not self.playing:
            return
        if self.loopback_frame_count < (self.loopback_samplerate or 1):
            return
        if self.loopback_peak_db <= -80.0:
            self.status_var.set("Speaker loopback is near silent")

    def _sync_monitor_button(self) -> None:
        if not hasattr(self, "capture_button"):
            return

        if self.spectrogram_source in {"capture", "return"}:
            if self.spectrogram_source == "return":
                text = "Stop 2.1 return" if self.capture_active else "Start 2.1 return"
            else:
                text = "Stop input" if self.capture_active else "Start input"
            self.capture_button.configure(text=text, state=tk.NORMAL)
        elif self.spectrogram_source == "loopback":
            text = "Stop loopback" if self.loopback_active else "Start loopback"
            self.capture_button.configure(text=text, state=tk.NORMAL)
        else:
            self.capture_button.configure(state=tk.DISABLED)

    @staticmethod
    def _signal_db(samples: np.ndarray) -> tuple[float, float]:
        if samples.size == 0:
            return -120.0, -120.0

        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(values)))
        rms = float(np.sqrt(np.mean(values * values)))
        return float(20.0 * np.log10(peak + 1e-10)), float(20.0 * np.log10(rms + 1e-10))

    @staticmethod
    def _create_spek_colormap() -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(
            "spek_classic",
            SPEK_COLOR_STOPS,
            N=4096,
        )

    def _style_spek_axes(self) -> None:
        self.ax.set_facecolor("#000000")
        self.ax.title.set_color(TEXT_PRIMARY)
        self.ax.xaxis.label.set_color(TEXT_MUTED)
        self.ax.yaxis.label.set_color(TEXT_MUTED)
        self.ax.tick_params(colors=TEXT_MUTED, which="both", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(PANEL_BORDER)
        self.ax.grid(color=PANEL_SUBTLE_BG, linewidth=0.45, alpha=0.62)
        self.ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4))
        self.ax.yaxis.set_major_locator(MaxNLocator(nbins=9, min_n_ticks=4))

    def _style_spek_colorbar(self) -> None:
        self.colorbar.ax.set_facecolor(GRAPH_BG)
        self.colorbar.outline.set_edgecolor(PANEL_BORDER)
        self.colorbar.set_label("")
        self.colorbar.set_ticks(COLORBAR_DB_TICKS)
        self.colorbar.set_ticklabels(COLORBAR_DB_LABELS)
        self.colorbar.ax.tick_params(colors=TEXT_PRIMARY, labelsize=8)

    # ---------- File loading ----------

    @staticmethod
    def _validate_audio_file_path(filename: str) -> Path:
        """Resolve and reject unsupported or oversized user-selected files."""
        try:
            path = Path(filename).expanduser().resolve(strict=True)
        except OSError as exc:
            raise ValueError("Audio file could not be found.") from exc

        if not path.is_file():
            raise ValueError("Selected path is not a file.")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise ValueError(f"Unsupported audio file type. Allowed: {allowed}.")

        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ValueError("Audio file metadata could not be read.") from exc

        if size <= 0:
            raise ValueError("Audio file is empty.")
        if size > MAX_AUDIO_FILE_BYTES:
            raise ValueError(
                "Audio file is too large to load safely "
                f"({format_bytes(size)} > {format_bytes(MAX_AUDIO_FILE_BYTES)})."
            )

        return path

    @staticmethod
    def _validate_audio_info(path: Path, info: Any) -> None:
        """Check decoder metadata before allocating the playback buffer."""
        frames = max(0, int(getattr(info, "frames", 0) or 0))
        channels = max(0, int(getattr(info, "channels", 0) or 0))
        samplerate = max(0, int(getattr(info, "samplerate", 0) or 0))
        playback_channels = min(channels, MAX_PLAYBACK_CHANNELS)

        if frames <= 0:
            raise ValueError("Audio file contains no decodable frames.")
        if channels <= 0:
            raise ValueError("Audio file contains no audio channels.")
        if channels > MAX_AUDIO_CHANNELS:
            raise ValueError(
                f"Audio file has too many channels to load safely ({channels} > {MAX_AUDIO_CHANNELS})."
            )
        if samplerate <= 0 or samplerate > MAX_AUDIO_SAMPLE_RATE:
            raise ValueError(f"Audio file sample rate is outside the safe range ({samplerate:,} Hz).")

        decoded_bytes = frames * playback_channels * np.dtype(np.float32).itemsize
        if decoded_bytes > MAX_DECODED_AUDIO_BYTES:
            raise ValueError(
                f"{path.name} would decode to about {format_bytes(decoded_bytes)}, "
                f"above the safe limit of {format_bytes(MAX_DECODED_AUDIO_BYTES)}."
            )

    @staticmethod
    def _read_playback_audio(audio_file: Any) -> tuple[np.ndarray, int]:
        """Read bounded blocks and keep only channels used for playback."""
        frames = int(getattr(audio_file, "frames", 0) or 0)
        channels = int(getattr(audio_file, "channels", 0) or 0)
        samplerate = int(getattr(audio_file, "samplerate", 0) or 0)
        playback_channels = min(channels, MAX_PLAYBACK_CHANNELS)
        data = np.empty((frames, playback_channels), dtype=np.float32)

        offset = 0
        while offset < frames:
            block = audio_file.read(
                min(AUDIO_LOAD_BLOCK_FRAMES, frames - offset),
                dtype="float32",
                always_2d=True,
            )
            if block.size == 0:
                break

            block_frames = int(block.shape[0])
            data[offset : offset + block_frames] = block[:, :playback_channels]
            offset += block_frames

        if offset < frames:
            data = data[:offset]

        return data, samplerate

    def open_file(self) -> None:
        dialog_options: dict[str, Any] = {
            "title": "Open audio file",
            "filetypes": [
                ("Audio files", "*.wav *.flac *.ogg *.aiff *.aif *.mp3"),
                ("All files", "*.*"),
            ],
        }
        if self.default_music_dir.exists():
            dialog_options["initialdir"] = str(self.default_music_dir)

        filename = filedialog.askopenfilename(**dialog_options)
        if not filename:
            return
        try:
            audio_path = self._validate_audio_file_path(filename)
        except ValueError as exc:
            self.status_var.set("Audio file rejected")
            messagebox.showerror("Could not load audio", str(exc))
            return

        self._close_stream(abort=True)
        self.playing = False
        self.end_reached = False
        self.stream_finished = False
        self.pending_play_after_resample = False

        self.loading = True
        self.load_token += 1
        self.resample_token += 1
        token = self.load_token

        self.open_button.configure(state=tk.DISABLED)
        self.play_button.configure(text="Play", state=tk.DISABLED)
        self.stop_button.configure(state=tk.DISABLED)
        self.file_var.set(f"Loading {audio_path.name}...")
        self.status_var.set("Loading audio in background")

        threading.Thread(
            target=self._load_audio_worker,
            args=(token, str(audio_path)),
            daemon=True,
        ).start()

    def _load_audio_worker(self, token: int, filename: str) -> None:
        """Decode audio off the GUI thread and return results through a queue."""
        try:
            audio_path = self._validate_audio_file_path(filename)
            with sf.SoundFile(str(audio_path), mode="r") as audio_file:
                self._validate_audio_info(audio_path, audio_file)
                data, samplerate = self._read_playback_audio(audio_file)
        except Exception as exc:  # noqa: BLE001 - user-facing error dialog
            self.load_result_queue.put(("error", token, filename, str(exc)))
            return

        if data.size == 0:
            self.load_result_queue.put(
                ("error", token, filename, "The file contains no audio.")
            )
            return

        data = np.nan_to_num(data, copy=False)
        if data.shape[1] == 1:
            mono = data[:, 0]
        else:
            mono = data.mean(axis=1, dtype=np.float32)

        self.load_result_queue.put(
            ("ok", token, filename, data, int(samplerate), mono)
        )

    def _handle_loaded_file(
        self,
        filename: str,
        data: np.ndarray,
        samplerate: int,
        mono: np.ndarray,
    ) -> None:
        path = Path(filename)
        source_key = (str(path), samplerate, len(data))
        with self.state_lock:
            self.audio = data
            self.mono = mono
            self.samplerate = samplerate
            self.playback_audio = data
            self.playback_samplerate = samplerate
            self.playback_source_key = source_key
            self.frame_index = 0

        self.path = path
        self.end_reached = False
        self.stream_finished = False
        self.playing = False
        self.resampling = False
        self.pending_play_after_resample = False
        self.open_button.configure(state=tk.NORMAL)
        self.play_button.configure(text="Play", state=tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL)

        duration = len(data) / float(samplerate)
        self.position_scale.configure(to=max(duration, 0.001))
        self.position_var.set(0.0)
        self.time_var.set(f"0:00 / {format_time(duration)}")
        self.file_var.set(f"{self.path.name} - {samplerate:,} Hz, {data.shape[1]} channel(s)")
        self.status_var.set("Loaded")

        self._reset_visual_buffer()
        self._reset_spectrogram_image()
        self._request_canvas_draw()

        target_rate = self._selected_output_samplerate()
        if target_rate != samplerate:
            self._start_playback_resample(target_rate, auto_play=False)

    # ---------- Playback ----------

    def _source_duration_seconds(self) -> float:
        if self.audio is None or self.samplerate is None:
            return 0.0
        return len(self.audio) / float(self.samplerate)

    def _current_playback_rate(self) -> int:
        return int(self.playback_samplerate or self.samplerate or 44_100)

    def _selected_output_samplerate(self) -> int:
        fallback = int(self.samplerate or self.playback_samplerate or 48_000)
        try:
            device_info = sd.query_devices(self._selected_output_device(), "output")
            return max(1, int(round(float(device_info["default_samplerate"]))))
        except Exception:
            return max(1, fallback)

    @staticmethod
    def _resample_audio_array(
        audio: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        """Resample playback audio when the output device requires it."""
        source_rate = int(source_rate)
        target_rate = int(target_rate)
        if source_rate == target_rate:
            return np.ascontiguousarray(audio, dtype=np.float32)

        try:
            soxr = import_module("soxr")
        except ImportError as exc:
            raise RuntimeError(
                "Install soxr to play files whose sample rate does not match "
                "the selected output device."
            ) from exc

        resampled = soxr.resample(audio, source_rate, target_rate, quality="HQ")
        return np.ascontiguousarray(resampled, dtype=np.float32)

    def _start_playback_resample(self, target_rate: int, *, auto_play: bool) -> bool:
        """Start background resampling and preserve the current play position."""
        if self.audio is None or self.samplerate is None:
            return False

        if self.resampling:
            self.pending_play_after_resample = self.pending_play_after_resample or auto_play
            if auto_play:
                self.play_button.configure(text="Preparing...", state=tk.DISABLED)
            self.status_var.set("Preparing playback rate")
            return False

        with self.state_lock:
            current_rate = self._current_playback_rate()
            current_seconds = self.frame_index / float(current_rate)
            source_audio = self.audio
            source_rate = self.samplerate
            source_key = (str(self.path) if self.path else None, source_rate, len(source_audio))

        if source_rate == target_rate:
            with self.state_lock:
                self.playback_audio = source_audio
                self.playback_samplerate = source_rate
                self.playback_source_key = source_key
                self.frame_index = min(
                    int(current_seconds * source_rate),
                    len(source_audio),
                )
            return True

        self.resampling = True
        self.pending_play_after_resample = auto_play
        self.resample_token += 1
        token = self.resample_token
        self.status_var.set(
            f"Preparing playback: {source_rate:,} Hz -> {target_rate:,} Hz"
        )
        if auto_play:
            self.play_button.configure(text="Preparing...", state=tk.DISABLED)

        threading.Thread(
            target=self._resample_audio_worker,
            args=(token, source_key, source_audio, source_rate, target_rate, current_seconds),
            daemon=True,
        ).start()
        return False

    def _resample_audio_worker(
        self,
        token: int,
        source_key: tuple[str | None, int, int],
        source_audio: np.ndarray,
        source_rate: int,
        target_rate: int,
        current_seconds: float,
    ) -> None:
        try:
            playback_audio = self._resample_audio_array(
                source_audio,
                source_rate,
                target_rate,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced by the GUI thread
            self.resample_result_queue.put(("error", token, str(exc)))
            return

        self.resample_result_queue.put(
            ("ok", token, source_key, playback_audio, target_rate, current_seconds)
        )

    def _playback_ready_for_output(self) -> bool:
        if self.audio is None or self.samplerate is None:
            return False

        target_rate = self._selected_output_samplerate()
        if (
            self.playback_audio is not None
            and self.playback_samplerate == target_rate
        ):
            return True

        return self._start_playback_resample(target_rate, auto_play=True)

    def toggle_play(self) -> None:
        if self.audio is None or self.samplerate is None:
            return

        if self.playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self.audio is None or self.samplerate is None:
            return
        if self.loading:
            self.status_var.set("Still loading audio")
            return
        if self.resampling:
            self.pending_play_after_resample = True
            self.play_button.configure(text="Preparing...", state=tk.DISABLED)
            self.status_var.set("Preparing playback rate")
            return
        if not self._playback_ready_for_output():
            return

        with self.state_lock:
            playback_audio = self.playback_audio
            playback_rate = self._current_playback_rate()
            total_frames = len(playback_audio) if playback_audio is not None else 0
            if self.frame_index >= total_frames:
                self.frame_index = 0
                self._reset_visual_buffer()

        if playback_audio is None or total_frames == 0:
            return

        self._close_stream(abort=True)
        self.end_reached = False
        self.stream_finished = False

        if self.spectrogram_source in {"capture", "return"} and not self.capture_active:
            if not self.start_capture():
                return
        if self.spectrogram_source == "loopback" and not self.loopback_active:
            if not self.start_loopback():
                return

        channels = int(playback_audio.shape[1])
        try:
            self.stream = sd.OutputStream(
                samplerate=playback_rate,
                channels=channels,
                dtype="float32",
                device=self._selected_output_device(),
                blocksize=1024,
                callback=self._audio_callback,
                finished_callback=self._stream_finished_callback,
            )
            self.stream.start()
        except Exception as exc:  # noqa: BLE001 - user-facing error dialog
            self.stream = None
            self.playing = False
            self.play_button.configure(text="Play")
            messagebox.showerror("Could not start playback", str(exc))
            return

        self.playing = True
        self.play_button.configure(text="Pause")
        self.status_var.set(f"Playing at {playback_rate:,} Hz")

    def pause(self) -> None:
        self._close_stream(abort=True)
        self.playing = False
        self.play_button.configure(text="Play")
        self.status_var.set("Paused")

    def stop(self) -> None:
        self._close_stream(abort=True)
        with self.state_lock:
            self.frame_index = 0
        self.playing = False
        self.pending_play_after_resample = False
        self.end_reached = False
        self.stream_finished = False
        self.play_button.configure(text="Play")
        self.position_var.set(0.0)
        self.status_var.set("Stopped")
        self._reset_visual_buffer()
        self._reset_spectrogram_image()
        self._request_canvas_draw()

    def _audio_callback(self, outdata: np.ndarray, frames: int, time, status) -> None:
        """Fill the PortAudio output buffer and enqueue data for the spectrogram."""
        del time  # Unused, but required by the sounddevice callback signature.

        if status:
            try:
                self.status_queue.put_nowait(str(status))
            except queue.Full:
                pass

        audio = self.playback_audio
        if audio is None:
            outdata.fill(0)
            raise sd.CallbackStop

        queue_player_visuals = self.spectrogram_source == "player"
        visual_chunks: list[np.ndarray] = []
        outdata.fill(0)

        with self.state_lock:
            total_frames = len(audio)
            start = min(self.frame_index, total_frames)

            if self.loop_enabled and total_frames:
                write_index = 0
                cursor = start
                while write_index < frames:
                    if cursor >= total_frames:
                        cursor = 0
                    take = min(frames - write_index, total_frames - cursor)
                    outdata[write_index : write_index + take] = audio[cursor : cursor + take]
                    if queue_player_visuals:
                        visual_chunks.append(audio[cursor : cursor + take])
                    cursor += take
                    write_index += take
                self.frame_index = cursor % total_frames
                self.end_reached = False
            else:
                end = min(start + frames, total_frames)
                if end > start:
                    outdata[: end - start] = audio[start:end]
                    if queue_player_visuals:
                        visual_chunks.append(audio[start:end])
                self.frame_index = end
                self.end_reached = end >= total_frames

        if self.volume != 1.0:
            outdata *= self.volume

        if visual_chunks:
            visual_chunk = self._visual_channel(np.concatenate(visual_chunks, axis=0))
            self._set_player_meter(visual_chunk)
            try:
                # Copy because the GUI thread consumes this later.
                self.visual_queue.put_nowait(visual_chunk.copy())
            except queue.Full:
                # Drop visual data rather than risking audio glitches.
                pass

        if self.end_reached and not self.loop_enabled:
            raise sd.CallbackStop

    def _stream_finished_callback(self) -> None:
        # Called from a non-GUI thread; the Tk loop will react to these flags.
        self.stream_finished = True

    def _close_stream(self, *, abort: bool) -> None:
        stream = self.stream
        self.stream = None
        if stream is None:
            return

        try:
            if stream.active:
                if abort:
                    stream.abort()
                else:
                    stream.stop()
        except Exception:
            pass

        try:
            stream.close()
        except Exception:
            pass

    # ---------- Live input capture ----------

    def toggle_capture(self) -> None:
        if self.spectrogram_source == "loopback":
            if self.loopback_active:
                self.stop_loopback()
            else:
                self.start_loopback()
            return

        if self.spectrogram_source in {"capture", "return"}:
            if self.capture_active:
                self.stop_capture()
            else:
                self.start_capture()
            return

        self.spectrogram_source_var.set("Sound card input")
        self._on_spectrogram_source_changed()
        self.start_capture()

    def _open_capture_stream(
        self, device_index: int | None
    ) -> tuple[sd.InputStream, int, int]:
        """Create a direct input stream using the selected/custom live rate."""
        device_info = sd.query_devices(device_index, "input")
        max_channels = max(1, int(device_info["max_input_channels"]))
        channels = min(max_channels, 8)
        default_samplerate = int(device_info["default_samplerate"] or self.samplerate or 48_000)
        samplerate = self._live_samplerate_for_default(default_samplerate)
        stream = sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
            device=device_index,
            blocksize=self._live_block_frames(samplerate),
            callback=self._capture_callback,
        )
        stream.start()
        return stream, samplerate, channels

    def start_capture(self) -> bool:
        self.stop_capture()

        device_index = self._selected_input_device()
        selected_label = self._label_for_input_device(device_index) or "System default"
        try:
            live_rate_status = self._live_samplerate_status()
        except ValueError as exc:
            messagebox.showerror("Invalid live sample rate", str(exc))
            return False
        if self.spectrogram_source == "return":
            block_reason = self._return_input_block_reason(selected_label)
            if block_reason is not None:
                messagebox.showerror(
                    "Select a 2.1 hardware return input",
                    (
                        f"{selected_label}\n\n{block_reason}\n\n"
                        "Choose the line-level USB interface receiving your front-left, "
                        "front-right, and sub/LFE soundcard outputs."
                    ),
                )
                return False
        attempted_labels = [selected_label]

        try:
            stream, samplerate, channels = self._open_capture_stream(device_index)
        except Exception as exc:  # noqa: BLE001 - user-facing error dialog
            first_error = exc
            stream = None
            samplerate = 0
            channels = 0
            for fallback_label in self._capture_fallback_labels(device_index):
                if (
                    self.spectrogram_source == "return"
                    and self._return_input_block_reason(fallback_label) is not None
                ):
                    continue
                attempted_labels.append(fallback_label)
                fallback_index = self.input_device_map[fallback_label]
                try:
                    stream, samplerate, channels = self._open_capture_stream(fallback_index)
                except Exception:
                    continue
                self.input_device_var.set(fallback_label)
                self.status_var.set(f"Capture fallback: {fallback_label}")
                break

            if stream is None:
                self.capture_stream = None
                self.capture_active = False
                attempted = "\n".join(f"- {label}" for label in attempted_labels)
                messagebox.showerror(
                    "Could not start capture input",
                    f"{first_error}\n\nTried:\n{attempted}",
                )
                return False

        self.capture_stream = stream

        self.capture_active = True
        self.capture_samplerate = samplerate
        self.capture_frame_count = 0
        self.capture_channels = channels
        self.capture_peak_db = self.db_floor
        self.capture_rms_db = self.db_floor
        self._reset_visual_buffer()
        self._reset_spectrogram_image()
        self._update_plot_title()
        self._update_meter_text()
        if self.spectrogram_source == "return":
            self.capture_button.configure(text="Stop 2.1 return", state=tk.NORMAL)
            if channels < 3:
                self.status_var.set("2.1 return needs a 3+ input interface; missing channels are muted")
            else:
                self.status_var.set(
                    f"Capturing 2.1 return at {samplerate:,} Hz: {self._return_channel_summary()}"
                )
        else:
            self.capture_button.configure(text="Stop input", state=tk.NORMAL)
            self.status_var.set(f"Capturing sound card input at {samplerate:,} Hz ({live_rate_status})")
        return True

    def stop_capture(self) -> None:
        self._close_capture_stream()
        self.capture_active = False
        self.capture_samplerate = None
        self.capture_frame_count = 0
        self.capture_channels = 0
        self.capture_peak_db = self.db_floor
        self.capture_rms_db = self.db_floor
        if hasattr(self, "capture_button"):
            if self.spectrogram_source == "return":
                self.capture_button.configure(text="Start 2.1 return")
            else:
                self.capture_button.configure(text="Start input")
        if hasattr(self, "meter_var"):
            self._update_meter_text()

    def start_loopback(self) -> bool:
        self.stop_loopback()

        if sc is None:
            messagebox.showerror(
                "Could not start speaker loopback",
                "The soundcard module is not installed.",
            )
            return False

        loopback_device = self._selected_loopback_device()
        if loopback_device is None:
            messagebox.showerror(
                "Could not start speaker loopback",
                "No soundcard loopback device is selected.",
            )
            return False

        channels = min(max(1, int(getattr(loopback_device, "channels", 2))), 8)
        try:
            samplerate = self._live_samplerate_for_default(
                self._loopback_default_samplerate(loopback_device)
            )
        except ValueError as exc:
            messagebox.showerror("Invalid live sample rate", str(exc))
            return False
        block_frames = self._loopback_block_frames(samplerate)
        self.loopback_stop_event.clear()
        self.loopback_active = True
        self.loopback_samplerate = samplerate
        self.loopback_frame_count = 0
        self.loopback_channels = channels
        self.loopback_peak_db = self.db_floor
        self.loopback_rms_db = self.db_floor
        self._reset_visual_buffer()
        self._reset_spectrogram_image()
        self._update_plot_title()
        self._update_meter_text()
        self.capture_button.configure(text="Stop loopback", state=tk.NORMAL)

        self.loopback_thread = threading.Thread(
            target=self._loopback_worker,
            args=(loopback_device, samplerate, channels, block_frames),
            daemon=True,
        )
        self.loopback_thread.start()
        self.status_var.set(f"Capturing speaker loopback at {samplerate:,} Hz")
        return True

    def stop_loopback(self) -> None:
        self.loopback_stop_event.set()
        thread = self.loopback_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self.loopback_thread = None
        self.loopback_active = False
        self.loopback_samplerate = None
        self.loopback_frame_count = 0
        self.loopback_channels = 0
        self.loopback_peak_db = self.db_floor
        self.loopback_rms_db = self.db_floor
        if hasattr(self, "capture_button") and self.spectrogram_source == "loopback":
            self.capture_button.configure(text="Start loopback", state=tk.NORMAL)
        if hasattr(self, "meter_var"):
            self._update_meter_text()

    def _loopback_worker(
        self, loopback_device: Any, samplerate: int, channels: int, block_frames: int
    ) -> None:
        """Record speaker loopback in a worker thread and queue visual chunks."""
        com_initialized = False
        try:
            com_initialized = self._initialize_com_for_thread()
            with loopback_device.recorder(
                samplerate=samplerate,
                channels=channels,
                blocksize=block_frames,
            ) as recorder:
                while not self.loopback_stop_event.is_set():
                    data = recorder.record(numframes=block_frames)
                    samples = np.asarray(data, dtype=np.float32)
                    if samples.size == 0:
                        continue
                    if self._window_interaction_active():
                        self.loopback_frame_count += self._visual_frame_count(samples)
                        continue
                    visual_chunk = self._visual_channel(samples)
                    self.loopback_frame_count += self._visual_frame_count(visual_chunk)
                    self._set_loopback_meter(visual_chunk)
                    try:
                        self.visual_queue.put_nowait(visual_chunk.copy())
                    except queue.Full:
                        pass
        except Exception as exc:  # noqa: BLE001 - reflected in the GUI status label
            try:
                self.status_queue.put_nowait(f"Loopback: {exc}")
            except queue.Full:
                pass
        finally:
            if com_initialized:
                self._uninitialize_com_for_thread()
            self.loopback_active = False

    @staticmethod
    def _initialize_com_for_thread() -> bool:
        if sys.platform != "win32":
            return False

        coinit_multithreaded = 0x0
        rpc_e_changed_mode = 0x80010106
        result = ctypes.windll.ole32.CoInitializeEx(None, coinit_multithreaded)
        hresult = result & 0xFFFFFFFF
        if hresult in (0, 1):
            return True
        if hresult == rpc_e_changed_mode:
            return False
        raise OSError(f"CoInitializeEx failed with HRESULT 0x{hresult:08X}")

    @staticmethod
    def _uninitialize_com_for_thread() -> None:
        if sys.platform == "win32":
            ctypes.windll.ole32.CoUninitialize()

    def _capture_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        del time  # Unused, but required by the sounddevice callback signature.

        if status:
            try:
                self.status_queue.put_nowait(str(status))
            except queue.Full:
                pass

        self.capture_frame_count += frames
        if indata.size == 0:
            return
        if self._window_interaction_active():
            return

        visual_chunk = self._visual_channel(indata)
        self._set_capture_meter(visual_chunk)
        try:
            self.visual_queue.put_nowait(visual_chunk.copy())
        except queue.Full:
            pass

    def _close_capture_stream(self) -> None:
        stream = self.capture_stream
        self.capture_stream = None
        if stream is None:
            return

        try:
            if stream.active:
                stream.stop()
        except Exception:
            pass

        try:
            stream.close()
        except Exception:
            pass

    # ---------- Seeking ----------

    def _on_slider_press(self, _event) -> None:
        if self.audio is not None:
            self.dragging_slider = True

    def _on_slider_release(self, _event) -> None:
        if self.audio is None or self.samplerate is None:
            self.dragging_slider = False
            return

        self.seek(self.position_var.get())
        self.dragging_slider = False

    def _on_slider_move(self, value: str) -> None:
        if self.audio is None or self.samplerate is None or not self.dragging_slider:
            return

        duration = self._source_duration_seconds()
        self.time_var.set(f"{format_time(float(value))} / {format_time(duration)}")

    def seek(self, seconds: float) -> None:
        if self.audio is None or self.samplerate is None:
            return

        duration = self._source_duration_seconds()
        seconds = min(max(0.0, float(seconds)), duration)
        playback_rate = self._current_playback_rate()
        playback_frames = len(self.playback_audio) if self.playback_audio is not None else 0
        with self.state_lock:
            self.frame_index = min(int(seconds * playback_rate), playback_frames)

        self.end_reached = False
        self.stream_finished = False
        self._reset_visual_buffer()
        self._reset_spectrogram_image()
        self.status_var.set("Playing" if self.playing else "Seeked")

    def seek_relative(self, seconds: float) -> None:
        if self.audio is None or self.samplerate is None:
            return

        with self.state_lock:
            current_frame = self.frame_index
            playback_rate = self._current_playback_rate()
        self.seek(current_frame / float(playback_rate) + seconds)

    # ---------- Spectrogram ----------

    def _visual_samplerate(self) -> int:
        if self.spectrogram_source in {"capture", "return"} and self.capture_samplerate is not None:
            return self.capture_samplerate
        if self.spectrogram_source == "loopback" and self.loopback_samplerate is not None:
            return self.loopback_samplerate
        return self.playback_samplerate or self.samplerate or 44_100

    def _reset_visual_buffer(self) -> None:
        """Clear queued visual data and allocate a ring for the current view."""
        while True:
            try:
                self.visual_queue.get_nowait()
            except queue.Empty:
                break

        self.spectrogram_render_token += 1
        samplerate = self._visual_samplerate()
        ring_len = max(int(self.window_seconds * samplerate), self.n_fft + self.hop_length)
        self.ring_channels = max(1, int(getattr(self, "ring_channels", 1)))
        self.ring = np.zeros((ring_len, self.ring_channels), dtype=np.float32)
        self.ring_index = 0
        self.spectrogram_dirty = True

    def _spectrogram_hop_for_length(self, sample_count: int) -> int:
        available = max(1, int(sample_count) - self.n_fft)
        hop_for_display = max(1, (available + self.max_spectrogram_columns - 1) // self.max_spectrogram_columns)
        return max(self.hop_length, hop_for_display)

    @staticmethod
    def _visual_frame_count(samples: np.ndarray) -> int:
        values = np.asarray(samples)
        if values.ndim == 0:
            return 0
        return int(values.shape[0])

    def _reset_spectrogram_image(self) -> None:
        self.spectrogram_render_token += 1
        samplerate = self._visual_samplerate()
        max_hz = min(self.display_max_hz, samplerate / 2.0)
        ring_len = max(int(self.window_seconds * samplerate), self.n_fft + self.hop_length)
        hop_length = self._spectrogram_hop_for_length(ring_len)
        frame_count = max(1, 1 + (ring_len - self.n_fft) // hop_length)
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / samplerate)
        freq_bins = max(1, int(np.count_nonzero(freqs <= max_hz)))

        blank = np.full((freq_bins, frame_count), self.db_floor, dtype=np.float32)
        self.image.set_data(blank)
        self.image.set_extent((0.0, self.window_seconds, 0.0, max_hz / 1000.0))
        self.image.set_clim(self.db_floor, 0.0)
        self.ax.set_ylim(0.0, max_hz / 1000.0)
        self.ax.set_xlim(0.0, self.window_seconds)
        self._style_spek_axes()
        self._style_spek_colorbar()
        self.spectrogram_dirty = True

    def _append_to_ring(self, samples: np.ndarray) -> None:
        """Append new visual samples into the fixed-size scrolling ring."""
        if samples.size == 0:
            return

        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)
        if values.size == 0 or values.shape[0] == 0:
            return

        channels = max(1, int(values.shape[1]))
        if self.ring is None:
            self.ring_channels = channels
            samplerate = self._visual_samplerate()
            ring_len = max(int(self.window_seconds * samplerate), self.n_fft + self.hop_length)
            self.ring = np.zeros((ring_len, channels), dtype=np.float32)
            self.ring_index = 0

        if self.ring.ndim != 2 or channels != self.ring_channels:
            self.ring_channels = channels
            samplerate = self._visual_samplerate()
            ring_len = max(int(self.window_seconds * samplerate), self.n_fft + self.hop_length)
            self.ring = np.zeros((ring_len, channels), dtype=np.float32)
            self.ring_index = 0

        n = int(values.shape[0])
        size = int(self.ring.shape[0])

        if n >= size:
            self.ring[:] = values[-size:]
            self.ring_index = 0
            self.spectrogram_dirty = True
            return

        end = self.ring_index + n
        if end <= size:
            self.ring[self.ring_index : end] = values
        else:
            split = size - self.ring_index
            self.ring[self.ring_index :] = values[:split]
            self.ring[: end % size] = values[split:]

        self.ring_index = end % size
        self.spectrogram_dirty = True

    def _ordered_ring_snapshot(self) -> np.ndarray:
        """Return the ring in chronological order for the FFT worker."""
        if self.ring is None:
            return np.zeros((self.n_fft + self.hop_length, 1), dtype=np.float32)
        if self.ring_index == 0:
            return self.ring.copy()
        return np.concatenate((self.ring[self.ring_index :], self.ring[: self.ring_index]))

    def _spectrogram_db(self, samples: np.ndarray) -> np.ndarray:
        samplerate = self._visual_samplerate()
        hop_length = self._spectrogram_hop_for_length(self._visual_frame_count(samples))
        db, _max_hz = self._spectrogram_db_for_params(
            samples,
            samplerate,
            self.n_fft,
            hop_length,
            self.db_floor,
            self.display_max_hz,
        )
        return db

    @staticmethod
    def _reverse_spectrogram_contrast(db: np.ndarray, db_floor: float) -> np.ndarray:
        values = np.asarray(db, dtype=np.float32)
        active_values = values[values > db_floor + 0.5]
        if active_values.size < 8:
            return values.astype(np.float32, copy=True)

        low = float(np.percentile(active_values, 10.0))
        high = float(np.percentile(active_values, 99.0))
        if high <= low + 1e-6:
            return values.astype(np.float32, copy=True)

        reversed_values = values.astype(np.float32, copy=True)
        active_mask = values >= low
        clipped = np.clip(values[active_mask], low, high)
        reversed_values[active_mask] = high + low - clipped
        reversed_values[values < low] = db_floor
        return np.clip(reversed_values, db_floor, 0.0).astype(np.float32, copy=False)

    @staticmethod
    def _apply_spek_display_gain(db: np.ndarray, db_floor: float) -> np.ndarray:
        return np.clip(db + SPEK_DISPLAY_GAIN_DB, db_floor, 0.0).astype(
            np.float32,
            copy=False,
        )

    def _should_invert_spectrogram_levels(self) -> bool:
        return bool(self.invert_live_var.get()) and self.spectrogram_source in {
            "capture",
            "return",
            "loopback",
        }

    @staticmethod
    def _spectrogram_db_for_params(
        samples: np.ndarray,
        samplerate: int,
        n_fft: int,
        hop_length: int,
        db_floor: float,
        display_max_hz: float,
    ) -> tuple[np.ndarray, float]:
        """Compute a dBFS spectrogram from one or more visual channels."""
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)

        if values.size == 0 or values.shape[0] < n_fft:
            return np.full((1, 1), db_floor, dtype=np.float32), min(display_max_hz, samplerate / 2.0)

        window = np.hanning(n_fft).astype(np.float32)
        magnitude_squared: np.ndarray | None = None
        active_channels = 0
        for channel_index in range(values.shape[1]):
            channel = values[:, channel_index]
            frames = np.lib.stride_tricks.sliding_window_view(channel, n_fft)[
                :: hop_length
            ]
            spectrum = np.fft.rfft(frames * window, axis=1)
            channel_magnitude = np.abs(spectrum) / (window.sum() / 2.0 + 1e-12)
            channel_power = channel_magnitude * channel_magnitude
            if magnitude_squared is None:
                magnitude_squared = channel_power
            else:
                magnitude_squared += channel_power
            active_channels += 1

        if magnitude_squared is None or active_channels == 0:
            return np.full((1, 1), db_floor, dtype=np.float32), min(display_max_hz, samplerate / 2.0)

        # Normalize by window sum so the color scale is approximately dBFS, then
        # combine channels after the FFT to avoid phase-cancelled spectrogram art.
        magnitude = np.sqrt(magnitude_squared / float(active_channels))
        db = 20.0 * np.log10(magnitude.T + 1e-10)

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / samplerate)
        max_hz = min(display_max_hz, samplerate / 2.0)
        db = db[freqs <= max_hz, :]
        return np.clip(db, db_floor, 0.0).astype(np.float32, copy=False), max_hz

    def _start_spectrogram_render(self) -> None:
        """Start a throttled background render when the ring has changed."""
        if self.ring is None or not self.spectrogram_dirty or self.spectrogram_rendering:
            return
        if self._window_interaction_active():
            return

        now = time.monotonic()
        if now - self.spectrogram_last_request < self.spectrogram_update_interval_seconds:
            return

        samples = self._ordered_ring_snapshot()
        samplerate = self._visual_samplerate()
        hop_length = self._spectrogram_hop_for_length(self._visual_frame_count(samples))
        invert_levels = self._should_invert_spectrogram_levels()
        self.spectrogram_last_request = now
        self.spectrogram_render_token += 1
        token = self.spectrogram_render_token
        self.spectrogram_rendering = True
        self.spectrogram_dirty = False

        threading.Thread(
            target=self._spectrogram_render_worker,
            args=(
                token,
                samples,
                samplerate,
                self.n_fft,
                hop_length,
                self.db_floor,
                self.display_max_hz,
                self.window_seconds,
                invert_levels,
            ),
            daemon=True,
        ).start()

    def _spectrogram_render_worker(
        self,
        token: int,
        samples: np.ndarray,
        samplerate: int,
        n_fft: int,
        hop_length: int,
        db_floor: float,
        display_max_hz: float,
        window_seconds: float,
        invert_levels: bool,
    ) -> None:
        """Run FFT work off-thread; `token` lets the GUI discard stale results."""
        try:
            db, max_hz = self._spectrogram_db_for_params(
                samples,
                samplerate,
                n_fft,
                hop_length,
                db_floor,
                display_max_hz,
            )
            db = self._apply_spek_display_gain(db, db_floor)
            if invert_levels:
                db = self._reverse_spectrogram_contrast(db, db_floor)
            result: tuple[Any, ...] = ("ok", token, db, max_hz, window_seconds)
        except Exception as exc:  # noqa: BLE001 - surfaced through the GUI loop
            result = ("error", token, str(exc))

        while True:
            try:
                self.spectrogram_result_queue.put_nowait(result)
                return
            except queue.Full:
                try:
                    self.spectrogram_result_queue.get_nowait()
                except queue.Empty:
                    return

    def _drain_load_results(self) -> None:
        while True:
            try:
                result = self.load_result_queue.get_nowait()
            except queue.Empty:
                return

            kind = result[0]
            token = result[1]
            if token != self.load_token:
                continue

            self.loading = False
            if kind == "error":
                _, _, filename, message = result
                self.open_button.configure(state=tk.NORMAL)
                self.play_button.configure(
                    text="Play",
                    state=tk.NORMAL if self.audio is not None else tk.DISABLED,
                )
                self.stop_button.configure(
                    state=tk.NORMAL if self.audio is not None else tk.DISABLED
                )
                self.file_var.set(f"Could not load {Path(filename).name}")
                self.status_var.set("Load failed")
                messagebox.showerror("Could not open audio file", message)
                continue

            _, _, filename, data, samplerate, mono = result
            self._handle_loaded_file(filename, data, samplerate, mono)

    def _drain_resample_results(self) -> None:
        while True:
            try:
                result = self.resample_result_queue.get_nowait()
            except queue.Empty:
                return

            kind = result[0]
            token = result[1]
            if token != self.resample_token:
                continue

            self.resampling = False
            should_play = self.pending_play_after_resample
            self.pending_play_after_resample = False
            self.play_button.configure(
                text="Play",
                state=tk.NORMAL if self.audio is not None else tk.DISABLED,
            )

            if kind == "error":
                _, _, message = result
                self.status_var.set("Playback preparation failed")
                if should_play:
                    messagebox.showerror("Could not prepare playback", message)
                continue

            _, _, source_key, playback_audio, target_rate, current_seconds = result
            current_key = (
                str(self.path) if self.path else None,
                self.samplerate or 0,
                len(self.audio) if self.audio is not None else 0,
            )
            if source_key != current_key:
                continue

            with self.state_lock:
                self.playback_audio = playback_audio
                self.playback_samplerate = int(target_rate)
                self.playback_source_key = source_key
                self.frame_index = min(
                    int(float(current_seconds) * int(target_rate)),
                    len(playback_audio),
                )

            if self.spectrogram_source == "player":
                self._reset_visual_buffer()
                self._reset_spectrogram_image()
                self._request_canvas_draw()

            if should_play:
                self.root.after_idle(self.play)
            else:
                self.status_var.set(f"Ready for playback at {int(target_rate):,} Hz")

    def _drain_spectrogram_results(self) -> None:
        if self._window_interaction_active():
            return

        while True:
            try:
                result = self.spectrogram_result_queue.get_nowait()
            except queue.Empty:
                return

            kind = result[0]
            token = result[1]
            self.spectrogram_rendering = False
            if token != self.spectrogram_render_token:
                continue

            if kind == "error":
                self.spectrogram_dirty = True
                try:
                    self.status_queue.put_nowait(f"Spectrogram: {result[2]}")
                except queue.Full:
                    pass
                continue

            _, _, db, max_hz, window_seconds = result
            self.image.set_data(db)
            self.image.set_extent((0.0, window_seconds, 0.0, max_hz / 1000.0))
            self.ax.set_xlim(0.0, window_seconds)
            self.ax.set_ylim(0.0, max_hz / 1000.0)
            self._request_canvas_draw()

    def _update_gui(self) -> None:
        """Drain worker queues, update widgets, and schedule the next UI tick."""
        window_busy = self._window_interaction_active()
        self._drain_load_results()
        self._drain_resample_results()
        if not window_busy:
            self._drain_spectrogram_results()

        while True:
            try:
                status = self.status_queue.get_nowait()
            except queue.Empty:
                break
            self.status_var.set(f"Audio: {status}")

        if window_busy:
            self.root.after(self.ui_interval_ms, self._update_gui)
            return

        # Drain audio chunks that the playback callback produced.
        for _ in range(MAX_VISUAL_CHUNKS_PER_UI_TICK):
            try:
                samples = self.visual_queue.get_nowait()
            except queue.Empty:
                break
            self._append_to_ring(samples)

        self._update_meter_text()
        self._sync_monitor_button()
        self._update_capture_silence_status()
        self._update_loopback_silence_status()

        if self.audio is not None and self.samplerate is not None:
            with self.state_lock:
                current_frame = self.frame_index
                playback_rate = self._current_playback_rate()
            duration = self._source_duration_seconds()
            current_seconds = min(current_frame / float(playback_rate), duration)

            if not self.dragging_slider:
                self.position_var.set(current_seconds)
            self.time_var.set(f"{format_time(current_seconds)} / {format_time(duration)}")
        elif self.capture_active and self.capture_samplerate:
            capture_seconds = self.capture_frame_count / float(self.capture_samplerate)
            label = "Return" if self.spectrogram_source == "return" else "Live input"
            self.time_var.set(f"{label} {format_time(capture_seconds)}")
        elif self.loopback_active and self.loopback_samplerate:
            loopback_seconds = self.loopback_frame_count / float(self.loopback_samplerate)
            self.time_var.set(f"Loopback {format_time(loopback_seconds)}")

        self._start_spectrogram_render()

        if self.stream_finished:
            self.stream_finished = False
            if self.end_reached:
                self._close_stream(abort=False)
                self.playing = False
                self.play_button.configure(text="Play")
                self.status_var.set("Finished")

        self._flush_deferred_canvas_draw()
        self.root.after(self.ui_interval_ms, self._update_gui)

    # ---------- Shutdown ----------

    def _on_close(self) -> None:
        self._restore_native_window_move_hook()
        self.stop_loopback()
        self._close_capture_stream()
        self._close_stream(abort=True)
        self.root.destroy()


def run_smoke_test() -> int:
    print(f"{APP_NAME} {APP_VERSION}")
    print(f"numpy {np.__version__}")
    print(f"sounddevice {sd.__version__}")
    print(f"soundfile {sf.__version__}")
    try:
        soxr = import_module("soxr")
    except ImportError:
        print("soxr unavailable")
        return 1
    print(f"soxr {soxr.__version__}")
    if sc is None:
        print("soundcard unavailable")
    else:
        print("soundcard available")

    sample = np.zeros((3840, 2), dtype=np.float32)
    resampled = LiveSpectrogramPlayer._resample_audio_array(sample, 384_000, 96_000)
    if resampled.shape != (960, 2):
        print(f"unexpected resample shape: {resampled.shape}")
        return 1
    t = np.arange(4096, dtype=np.float32) / 48_000.0
    sine = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)
    antiphase = np.column_stack((sine, -sine)).astype(np.float32)
    antiphase_db, _ = LiveSpectrogramPlayer._spectrogram_db_for_params(
        antiphase,
        48_000,
        1024,
        512,
        -120.0,
        20_000.0,
    )
    if float(np.max(antiphase_db)) < -12.0:
        print(f"phase-safe channel mix failed: {float(np.max(antiphase_db)):.1f} dB")
        return 1
    reverse_input = np.array(
        [[-120.0, -95.0, -90.0, -85.0, -80.0, -75.0, -70.0, -65.0, -60.0, -55.0]],
        dtype=np.float32,
    )
    reversed_db = LiveSpectrogramPlayer._reverse_spectrogram_contrast(reverse_input, -120.0)
    if float(reversed_db[0, 0]) != -120.0 or float(np.max(reversed_db)) >= -1.0:
        print(f"unsafe reversed dB mapping: {reversed_db}")
        return 1
    if LiveSpectrogramPlayer._loopback_block_frames(96_000) != 9600:
        print("unexpected loopback block size")
        return 1
    live_rate_cases = {
        "Auto": None,
        "8 kHz": 8_000,
        "11.025 kHz": 11_025,
        "22050": 22_050,
        "96000": 96_000,
        "96 kHz": 96_000,
        "44.1 kHz": 44_100,
        "192000 Hz": 192_000,
    }
    for label, expected_rate in live_rate_cases.items():
        actual_rate = LiveSpectrogramPlayer._parse_live_samplerate(label)
        if actual_rate != expected_rate:
            print(f"unexpected live sample rate parse: {label!r} -> {actual_rate}")
            return 1
    try:
        LiveSpectrogramPlayer._parse_live_samplerate("2 Hz")
    except ValueError:
        pass
    else:
        print("invalid live sample rate was accepted")
        return 1
    if COLORBAR_DB_LABELS != ("-120 dB", "-100 dB", "-80 dB", "-60 dB", "-40 dB", "-20 dB", "0 dB"):
        print(f"unexpected colorbar labels: {COLORBAR_DB_LABELS}")
        return 1
    if not app_resource_path(*APP_ICON_PATH).exists():
        print("missing app icon asset")
        return 1
    if format_bytes(1536) != "1.5 KiB":
        print("unexpected byte formatter output")
        return 1
    safe_info = type("AudioInfo", (), {"frames": 48_000, "channels": 2, "samplerate": 48_000})()
    LiveSpectrogramPlayer._validate_audio_info(Path("safe.wav"), safe_info)
    unsafe_rate_info = type("AudioInfo", (), {"frames": 48_000, "channels": 2, "samplerate": 1_000_000})()
    try:
        LiveSpectrogramPlayer._validate_audio_info(Path("unsafe.wav"), unsafe_rate_info)
    except ValueError:
        pass
    else:
        print("unsafe sample rate was accepted")
        return 1
    oversized_frames = (MAX_DECODED_AUDIO_BYTES // (2 * np.dtype(np.float32).itemsize)) + 1
    oversized_info = type(
        "AudioInfo",
        (),
        {"frames": oversized_frames, "channels": 2, "samplerate": 48_000},
    )()
    try:
        LiveSpectrogramPlayer._validate_audio_info(Path("oversized.wav"), oversized_info)
    except ValueError:
        pass
    else:
        print("oversized decoded audio was accepted")
        return 1
    multichannel_limit_info = type(
        "AudioInfo",
        (),
        {"frames": oversized_frames - 1, "channels": MAX_AUDIO_CHANNELS, "samplerate": 48_000},
    )()
    LiveSpectrogramPlayer._validate_audio_info(Path("wide.wav"), multichannel_limit_info)
    class FakeSoundFile:
        frames = 4
        channels = 3
        samplerate = 48_000

        def __init__(self) -> None:
            self.offset = 0
            self.samples = np.arange(12, dtype=np.float32).reshape(4, 3)

        def read(self, frame_count: int, *, dtype: str, always_2d: bool) -> np.ndarray:
            _ = (dtype, always_2d)
            end = min(self.offset + int(frame_count), self.frames)
            block = self.samples[self.offset : end]
            self.offset = end
            return block

    fake_file = FakeSoundFile()
    fake_data, fake_rate = LiveSpectrogramPlayer._read_playback_audio(fake_file)
    expected_fake_data = np.array([[0, 1], [3, 4], [6, 7], [9, 10]], dtype=np.float32)
    if fake_rate != 48_000 or not np.array_equal(fake_data, expected_fake_data):
        print("optimized audio reader returned unexpected data")
        return 1
    empty_ring_app = type("RingHarness", (), {"ring": None, "n_fft": 8, "hop_length": 2})()
    empty_ring = LiveSpectrogramPlayer._ordered_ring_snapshot(empty_ring_app)
    if empty_ring.shape != (10, 1):
        print(f"unexpected empty ring snapshot shape: {empty_ring.shape}")
        return 1
    wrapped_ring_app = type(
        "RingHarness",
        (),
        {
            "ring": np.arange(12, dtype=np.float32).reshape(6, 2),
            "ring_index": 2,
            "n_fft": 8,
            "hop_length": 2,
        },
    )()
    wrapped = LiveSpectrogramPlayer._ordered_ring_snapshot(wrapped_ring_app)
    if not np.array_equal(wrapped[:, 0], np.array([4, 6, 8, 10, 0, 2], dtype=np.float32)):
        print(f"unexpected wrapped ring order: {wrapped[:, 0]}")
        return 1
    cmap = LiveSpectrogramPlayer._create_spek_colormap()
    for stop, expected_hex in SPEK_COLOR_STOPS:
        actual_hex = to_hex(cmap(stop), keep_alpha=False)
        actual = tuple(int(actual_hex[index : index + 2], 16) for index in (1, 3, 5))
        expected = tuple(int(expected_hex[index : index + 2], 16) for index in (1, 3, 5))
        if any(abs(a - e) > 2 for a, e in zip(actual, expected)):
            print(f"unexpected Spek colour at {stop:.6f}: {actual_hex} != {expected_hex}")
            return 1
    print("smoke test ok")
    return 0


def main() -> None:
    if "--version" in sys.argv:
        print(f"{APP_NAME} {APP_VERSION}")
        return
    if "--smoke-test" in sys.argv:
        raise SystemExit(run_smoke_test())

    root = ctk.CTk()
    LiveSpectrogramPlayer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
