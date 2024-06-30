import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def save_audio(y, sr, output_path):
    sf.write(output_path, y, sr)

def generate_watermark(length, seed=42):
    np.random.seed(seed)
    return np.random.randn(length)

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def perceptual_weighting(audio, sr):
    # Apply a low-pass filter to the audio to simulate perceptual weighting
    filtered_audio = lowpass_filter(audio, 4000, sr)  # Filter frequencies above 4kHz
    return filtered_audio

def embed_watermark(audio, watermark, alpha=0.01):
    # Apply perceptual weighting to the watermark signal
    perceptual_watermark = perceptual_weighting(watermark, sr=len(audio))
    # Repeat the watermark signal to ensure it's longer than the audio signal
    watermark_signal = np.tile(perceptual_watermark, len(audio) // len(perceptual_watermark) + 1)
    # Trim the watermark signal to match the exact length of the audio signal
    watermark_signal = watermark_signal[:len(audio)]
    # Ensure the lengths match
    assert len(audio) == len(watermark_signal), "Length mismatch between audio and watermark signal"
    watermarked_audio = audio + alpha * watermark_signal
    return watermarked_audio

def main():
    input_directory = 'audio_watermark_env/data/original'
    output_directory = 'audio_watermark_env/data/watermarked'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            
            # Load the original audio file
            audio, sr = load_audio(input_path)
            print(f"Loaded {filename} with length {len(audio)}")
            
            # Generate a watermark
            watermark = generate_watermark(len(audio))
            
            # Embed the watermark into the audio
            watermarked_audio = embed_watermark(audio, watermark)
            print(f"Watermarked {filename} with length {len(watermarked_audio)}")
            
            # Save the watermarked audio file
            save_audio(watermarked_audio, sr, output_path)
            print(f"Watermarked audio saved to {output_path}")

if __name__ == "__main__":
    main()
