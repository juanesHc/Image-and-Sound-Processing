import librosa
from tkinter import Tk, filedialog

def selectAudio():
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter
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
