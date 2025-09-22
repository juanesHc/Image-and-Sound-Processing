import tkinter as tk
from tkinter import Button, Label, messagebox
from PIL import Image, ImageTk
import sounddevice as sd
import numpy as np
import librosa

import imageAnalysis as ia
import audioAnalysis as aa
from fusion import createFusionFigure   # Debe existir fusion.py


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Proyecto - Procesamiento de Imagen y Audio")

        # ==== Variables ====
        self.image = None
        self.image_rgb = None
        self.audio = None
        self.sr = None
        self.image_path = None
        self.audio_path = None

        # ==== UI ====
        self.img_label = Label(root, text="No se ha cargado ninguna imagen")
        self.img_label.pack(pady=10)

        self.audio_label = Label(root, text="No se ha cargado ningún audio")
        self.audio_label.pack(pady=10)

        Button(root, text="Cargar Imagen", command=self.load_image).pack(pady=5)
        Button(root, text="Cargar Audio", command=self.load_audio).pack(pady=5)
        Button(root, text="Analizar", command=self.analyze).pack(pady=5)

    # =====================
    # BOTONES
    # =====================
    def load_image(self):
        """Carga imagen y la muestra en el Label."""
        self.image_path = ia.selectImage()
        if self.image_path:
            img_bgr = ia.loadImage(self.image_path)
            if isinstance(img_bgr, str):
                self.img_label.config(text=img_bgr)
                return

            # Guardar en memoria
            self.image = img_bgr
            self.image_rgb = img_bgr[:, :, ::-1]  # BGR -> RGB

            # Mostrar en Tkinter
            pil_img = Image.fromarray(self.image_rgb).resize((300, 200))
            tk_img = ImageTk.PhotoImage(pil_img)
            self.img_label.config(image=tk_img, text="")
            self.img_label.image = tk_img

    def load_audio(self):
        """Carga audio, muestra ruta y lo reproduce."""
        self.audio_path = aa.selectAudio()
        if self.audio_path:
            try:
                self.audio, self.sr = aa.loadAudio(self.audio_path)
                self.audio_label.config(text=f"Audio cargado: {self.audio_path}")
                # Reproducción (no bloqueante)
                sd.play(self.audio, self.sr)
            except Exception as e:
                self.audio_label.config(text=f"Error: {str(e)}")

    def analyze(self):
        """Ejecuta el análisis de imagen, audio y genera figura combinada."""
        if self.image is None or self.audio is None:
            messagebox.showwarning("Advertencia", "Debe cargar imagen y audio antes de analizar.")
            return

        # ===== Imagen =====
        palette = ia.getDominantColors(self.image, n=5)
        ia.showColorPalette(palette)
        hist_data = ia.getColorHistogram(self.image)
        ia.generateHistogramColorsImage(hist_data)

        # ===== Audio =====
        duration = aa.getDuration(self.audio, self.sr)
        aa.plotWaveform(self.audio, self.sr)
        freqs, magnitude = aa.getSpectrum(self.audio, self.sr)
        mel_spec = aa.getSpectrogram(self.audio, self.sr)

        # Features extra para fusión
        hop_length = 512
        rms = librosa.feature.rms(y=self.audio, frame_length=2048, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr, hop_length=hop_length)[0]
        tempo, _ = librosa.beat.beat_track(y=self.audio, sr=self.sr)
        tempo = float(np.atleast_1d(tempo)[0])  # 🔑 Garantiza que sea float

        # ===== Fusión =====
        createFusionFigure(
            image=self.image_rgb,
            palette=palette,
            hist_data=hist_data,
            mel_spectrogram=mel_spec,
            rms=rms,
            centroid=centroid,
            sr=self.sr,
            hop_length=hop_length,
            duration=duration,
            tempo=tempo
        )

        messagebox.showinfo("Análisis completo", "Se generaron los análisis de imagen, audio y fusión.")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
