import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def plot_and_save_audio(audio, sr, title, filename):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.savefig(f'audio_watermark_env/data/pics/{filename}.png')
    plt.close()

def play_audio(audio, sr):
    sd.play(audio, sr)
    sd.wait()

def main():
    # Path to your directory containing audio files
    directory_path = 'audio_watermark_env/data/original'
    pics_directory = 'audio_watermark_env/data/pics'
    
    # Create pics directory if it doesn't exist
    if not os.path.exists(pics_directory):
        os.makedirs(pics_directory)
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return
    
    # Process each .wav file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            
            # Load audio file
            audio, sr = load_audio(file_path)
            
            # Plot and save audio file
            plot_and_save_audio(audio, sr, title=f"Audio Signal - {filename}", filename=filename)
            
            # Play audio file
            play_audio(audio, sr)
        else:
            print(f"Skipping non-wav file: {filename}")

if __name__ == "__main__":
    main()
