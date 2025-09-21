import tkinter as tk
from tkinter import Button, Label, filedialog
from PIL import Image, ImageTk
import sounddevice as sd

import imageAnalysis as ia
import audioAnalysis as aa


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Proyecto - Procesamiento de Imagen y Audio")

        # Variables de estado
        self.image = None
        self.audio = None
        self.sr = None
        self.image_path = None
        self.audio_path = None

        # Widgets
        self.img_label = Label(root, text="No se ha cargado ninguna imagen")
        self.img_label.pack(pady=10)

        self.audio_label = Label(root, text="No se ha cargado ningún audio")
        self.audio_label.pack(pady=10)

        Button(root, text="Cargar Imagen", command=self.load_image).pack(pady=5)
        Button(root, text="Cargar Audio", command=self.load_audio).pack(pady=5)
        Button(root, text="Analizar", command=self.analyze).pack(pady=5)

    # =====================
    # FUNCIONES DE BOTONES
    # =====================

    def load_image(self):
        self.image_path = ia.selectImage()
        if self.image_path:
            self.image = ia.loadImage(self.image_path)
            if isinstance(self.image, str):  # Error de carga
                self.img_label.config(text=self.image)
            else:
                # Mostrar imagen en Tkinter
                img_rgb = self.image[:, :, ::-1]
                pil_img = Image.fromarray(img_rgb)
                pil_img = pil_img.resize((300, 200))
                tk_img = ImageTk.PhotoImage(pil_img)
                self.img_label.config(image=tk_img, text="")
                self.img_label.image = tk_img

    def load_audio(self):
        self.audio_path = aa.selectAudio()
        if self.audio_path:
            try:
                self.audio, self.sr = aa.loadAudio(self.audio_path)
                self.audio_label.config(text=f"Audio cargado: {self.audio_path}")
                sd.play(self.audio, self.sr)
                sd.wait()
            except Exception as e:
                self.audio_label.config(text=f"Error: {str(e)}")

    def analyze(self):
        if self.image is None or self.audio is None:
            tk.messagebox.showwarning("Advertencia", "Debe cargar imagen y audio antes de analizar.")
            return

        # Análisis de imagen
        colors = ia.getDominantColors(self.image, n=5)
        ia.showColorPalette(colors)
        hist_data = ia.getColorHistogram(self.image)
        ia.generateHistogramColorsImage(hist_data)

        # Análisis de audio (ejemplo: duración)
        duration = len(self.audio) / self.sr
        self.audio_label.config(text=f"Audio cargado: {self.audio_path} (Duración: {duration:.2f} seg)")

        tk.messagebox.showinfo("Análisis completo", "Se generaron los análisis de imagen y audio.")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
