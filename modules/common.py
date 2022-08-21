import librosa
import numpy as np

# estimate frequency bandwidth of signal
def eval_bandwidth(
    signal: np.ndarray, sr: int, 
    time_stride: float = 0.01, n_fft: int = 512, threshold: int = -50) -> float:

    hop_length = int(sr * time_stride)
    spectrogram = np.mean(
        np.abs(
            librosa.stft(
                y=signal, n_fft=n_fft, hop_length=hop_length, window='blackmanharris',
                )) ** 2, axis=1
    )
    power_spectrum = librosa.power_to_db(S=spectrogram, ref=np.max, top_db=100)
    freqband: int = 0
    for idx in range(len(power_spectrum) - 1, -1, -1):
        if power_spectrum[idx] > threshold:
            freqband = int(idx / n_fft * sr)
            break

    return freqband