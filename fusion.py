import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def createFusionFigure(
    image,                # Imagen en BGR o RGB (se asume que ya la conviertes a RGB antes de pasarla)
    palette,              # Lista de colores dominantes [(R,G,B), ...]
    hist_data,            # Diccionario {"r": array, "g": array, "b": array}
    mel_spectrogram,      # Espectrograma en dB (salida de getSpectrogram)
    rms,                  # Array de RMS por frame
    centroid,             # Array de spectral centroid por frame
    sr,                   # Sample rate del audio
    hop_length,           # Hop length usado en RMS/centroid
    duration,             # Duración total del audio (seg)
    tempo,                # Tempo estimado (BPM)
    save_path="outputs/fusion_figure.png"
):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # --- Imagen Original ---
    axs[0,0].imshow(image)
    axs[0,0].axis("off")
    axs[0,0].set_title("Imagen Original")

    # --- Paleta de colores ---
    palette_img = np.zeros((100, 100 * len(palette), 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        palette_img[:, i*100:(i+1)*100, :] = color
    axs[0,1].imshow(palette_img)
    axs[0,1].axis("off")
    axs[0,1].set_title("Colores Dominantes")

    # --- Histograma RGB ---
    axs[0,2].plot(hist_data["r"], color="red", label="Rojo")
    axs[0,2].plot(hist_data["g"], color="green", label="Verde")
    axs[0,2].plot(hist_data["b"], color="blue", label="Azul")
    axs[0,2].set_title("Histograma de Colores")
    axs[0,2].legend()

    # --- Mel-Spectrogram ---
    img = librosa.display.specshow(
        mel_spectrogram,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        hop_length=hop_length,
        cmap="magma",
        ax=axs[1,0]
    )
    # Usar el objeto devuelto para la barra de color
    fig.colorbar(img, ax=axs[1,0], format="%+2.0f dB")

    # --- RMS + Spectral Centroid ---
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_norm = rms / rms.max()
    centroid_norm = (centroid - centroid.min()) / (centroid.max() - centroid.min())
    axs[1,1].plot(t, rms_norm, color="purple", label="RMS (norm.)")
    axs[1,1].plot(t, centroid_norm, color="orange", label="Centroid (norm.)")
    axs[1,1].set_xlabel("Tiempo (s)")
    axs[1,1].set_title("RMS & Spectral Centroid")
    axs[1,1].legend()

    # --- Métricas ---
    axs[1,2].axis("off")
    texto = (
        f"Duración: {duration:.2f} s\n"
        f"Tempo: {tempo:.1f} BPM\n"
        f"RMS medio: {rms.mean():.3f}\n"
        f"Centroid medio: {centroid.mean():.1f} Hz"
    )
    axs[1,2].text(0.1, 0.5, texto, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
