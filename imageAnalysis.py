import cv2
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog

def selectImage():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path


def loadImage(path):
    if not path:
        return("No se seleccionó ninguna imagen.")
    image = cv2.imread(path)
    if image is None:
        return("Error al cargar la imagen.")
    return image

def getDominantColors(image, n , reduce_factor=100):
    img_rgb = image[:, :, ::-1]
    img_reduced = (img_rgb // reduce_factor) * reduce_factor
    pixels = img_reduced.reshape(-1, 3)


    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    sorted_idx = np.argsort(-counts)
    dominant_colors = unique_colors[sorted_idx][:n]

    return [tuple(color) for color in dominant_colors]



def getColorHistogram(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten()
    return {
        "r": hist_r,
        "g": hist_g,
        "b": hist_b
        }


def generateHistogramColorsImage(hist_data, save_path="outputs/color_histogram.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(hist_data["r"], color="red", label="Rojo")
    plt.plot(hist_data["g"], color="green", label="Verde")
    plt.plot(hist_data["b"], color="blue", label="Azul")
    plt.legend()
    plt.title("Histograma de Colores")
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def showColorPalette(colors, save_path="outputs/palette.png"):

    palette = np.zeros((100, 100 * len(colors), 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        palette[:, i*100:(i+1)*100, :] = color

    plt.imshow(palette)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

