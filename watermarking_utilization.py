import os
import numpy as np
import librosa
import soundfile as sf
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def normalize_audio(audio):
    mean = np.mean(audio)
    std = np.std(audio)
    normalized_audio = (audio - mean) / std
    return normalized_audio, mean, std

def save_audio(y, sr, output_path):
    sf.write(output_path, y, sr)

def plot_and_save_audio(audio, sr, title, filename):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.savefig(filename)
    plt.close()
    return filename

def pad_or_truncate(audio, length):
    if len(audio) > length:
        return audio[:length]
    else:
        return np.pad(audio, (0, length - len(audio)), 'constant')

def add_watermark(audio, model, alpha=0.01):
    watermark_signal = model.predict(audio.reshape(1, -1, 1)).flatten()
    watermarked_audio = audio + alpha * watermark_signal
    return watermarked_audio

def process_audio_file(filename, input_directory, output_directory, pics_directory, model, max_length, log_file):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        
        try:
            start_time = time.time()
            
            # Load and normalize the audio file
            audio, sr = load_audio(input_path)
            load_time = time.time() - start_time
            
            normalized_audio, mean_before, std_before = normalize_audio(audio)
            normalize_time = time.time() - start_time - load_time
            
            # Pad or truncate the audio to the fixed length
            padded_audio = pad_or_truncate(normalized_audio, max_length)
            pad_time = time.time() - start_time - load_time - normalize_time
            
            # Add watermark to the audio
            watermarked_audio = add_watermark(padded_audio, model)
            watermark_time = time.time() - start_time - load_time - normalize_time - pad_time
            
            # Calculate mean and std after watermarking
            mean_after = np.mean(watermarked_audio)
            std_after = np.std(watermarked_audio)
            
            # Save the watermarked audio
            save_audio(watermarked_audio, sr, output_path)
            save_time = time.time() - start_time - load_time - normalize_time - pad_time - watermark_time
            
            # Plot and save original and watermarked audio
            original_plot_path = plot_and_save_audio(normalized_audio, sr, title=f"Original Audio Signal - {filename}", filename=f'{pics_directory}/original_{filename}.png')
            watermarked_plot_path = plot_and_save_audio(watermarked_audio, sr, title=f"Watermarked Audio Signal - {filename}", filename=f'{pics_directory}/watermarked_{filename}.png')
            plot_time = time.time() - start_time - load_time - normalize_time - pad_time - watermark_time - save_time
            
            total_time = time.time() - start_time
            
            log_file.write(f"Processed audio file: {input_path}\n")
            log_file.write(f"  Length: {len(audio)} samples\n")
            log_file.write(f"  Sample rate: {sr} Hz\n")
            log_file.write(f"  Mean before normalization: {mean_before:.6f}\n")
            log_file.write(f"  Std before normalization: {std_before:.6f}\n")
            log_file.write(f"  Mean after watermarking: {mean_after:.6f}\n")
            log_file.write(f"  Std after watermarking: {std_after:.6f}\n")
            log_file.write(f"  Load time: {load_time:.6f} seconds\n")
            log_file.write(f"  Normalize time: {normalize_time:.6f} seconds\n")
            log_file.write(f"  Pad/truncate time: {pad_time:.6f} seconds\n")
            log_file.write(f"  Watermark time: {watermark_time:.6f} seconds\n")
            log_file.write(f"  Save time: {save_time:.6f} seconds\n")
            log_file.write(f"  Plot time: {plot_time:.6f} seconds\n")
            log_file.write(f"  Total processing time: {total_time:.6f} seconds\n")
            log_file.write(f"  Watermarked audio saved to: {output_path}\n")
            log_file.write(f"  Original audio plot saved to: {original_plot_path}\n")
            log_file.write(f"  Watermarked audio plot saved to: {watermarked_plot_path}\n\n\n")
        except Exception as e:
            log_file.write(f"Failed to process audio file: {input_path}\n")
            log_file.write(f"  Error: {str(e)}\n")
    else:
        log_file.write(f"Skipping non-wav file: {filename}\n")

def main():
    # Load the trained model
    model_path = 'audio_watermark_env/audio_watermarking_model.h5'
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    
    # Paths to your audio files
    input_directory = 'audio_watermark_env/data/new_audio'
    output_directory = 'audio_watermark_env/data/results'
    pics_directory = 'audio_watermark_env/data/result_pics'
    log_file_path = 'audio_watermark_env/data/watermarking_utilization.txt'
    
    # Create output directories if they don't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(pics_directory):
        os.makedirs(pics_directory)
    
    max_length = 441000  # Length of the audio used during training

    with open(log_file_path, 'w') as log_file:
        log_file.write("Watermarking Utilization Process Log\n")
        
        for filename in os.listdir(input_directory):
            process_audio_file(filename, input_directory, output_directory, pics_directory, model, max_length, log_file)

if __name__ == "__main__":
    main()
