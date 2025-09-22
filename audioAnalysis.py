import librosa
from tkinter import Tk, filedialog
import numpy as np
import matplotlib.pyplot as plt

def selectAudio():
    root = Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(
        title="Selecciona un archivo de audio",
        filetypes=[("Archivos de audio", "*.wav *.mp3 *.flac *.ogg")]
    )
    return file_path


def loadAudio(path, sr=22050):
    if not path:
        raise ValueError("No se seleccionó ningún archivo de audio.")
    
    audio, sr = librosa.load(path, sr=sr)  
    return audio, sr

def plotWaveform(audio, sr):
    import matplotlib.pyplot as plt
    import numpy as np
    time = np.linspace(0, len(audio) / sr, num=len(audio))
    plt.figure(figsize=(8, 3))
    plt.plot(time, audio)
    plt.title("Forma de onda")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.tight_layout()
    plt.savefig("outputs/waveform.png")

def getDuration(audio, sr):
    return len(audio) / sr

def getSpectrum(audio, sr):

    N = len(audio)
    spectrum = np.fft.fft(audio)
    freqs = np.fft.fftfreq(N, d=1/sr)

    mask = freqs >= 0
    freqs = freqs[mask]
    magnitude = np.abs(spectrum[mask])

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, magnitude, color="purple")
    plt.title("Espectro de Frecuencias (FFT)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
    plt.tight_layout()
    plt.savefig("outputs/spectrum.png")
    plt.close()

    return freqs, magnitude

def getSpectrogram(audio, sr, save_path="outputs/audio_spectrogram.png"):
    stft = librosa.stft(audio)
    db_spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        db_spectrogram,
        sr=sr,
        x_axis='time',
        y_axis='hz',
        cmap='magma'
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Espectrograma")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return db_spectrogram