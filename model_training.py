import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime

def load_audio_file(file_path, target_sr=44100, max_length=441000):
    y, sr = librosa.load(file_path, sr=target_sr)
    if len(y) > max_length:
        y = y[:max_length]
    else:
        y = np.pad(y, (0, max_length - len(y)), 'constant')
    return y, sr

def normalize_audio(audio):
    return (audio - np.mean(audio)) / np.std(audio)

def preprocess_data(original_files, watermarked_files, max_length=441000):
    X, y = [], []
    for (original, sr_orig), (watermarked, sr_wm) in zip(original_files, watermarked_files):
        if sr_orig != sr_wm:
            print(f"Sample rates do not match for files: {sr_orig} != {sr_wm}")
            continue
        
        original = normalize_audio(original)
        watermarked = normalize_audio(watermarked)
        
        if len(original) > max_length:
            original = original[:max_length]
        else:
            original = np.pad(original, (0, max_length - len(original)), 'constant')
        
        if len(watermarked) > max_length:
            watermarked = watermarked[:max_length]
        else:
            watermarked = np.pad(watermarked, (0, max_length - len(watermarked)), 'constant')
        
        X.append(original.reshape(-1, 1))
        y.append(watermarked.reshape(-1, 1))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_autoencoder(input_shape):
    input_audio = layers.Input(shape=input_shape)
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(input_audio)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(8, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(2, padding='same')(x)
    
    x = layers.Conv1D(8, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    decoded = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x)
    
    autoencoder = models.Model(input_audio, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def main():
    # Define the directory paths
    original_directory = 'audio_watermark_env/data/original'
    watermarked_directory = 'audio_watermark_env/data/watermarked'
    result_file_path = 'audio_watermark_env/model_training_results.txt'
    
    # Initialize result file
    with open(result_file_path, 'w') as result_file:
        result_file.write(f"Model Training Results\n")
        result_file.write(f"Start Time: {datetime.now()}\n\n")
    
    # Get list of audio files
    original_files = [os.path.join(original_directory, f) for f in os.listdir(original_directory) if f.endswith('.wav')]
    watermarked_files = [os.path.join(watermarked_directory, f) for f in os.listdir(watermarked_directory) if f.endswith('.wav')]
    
    if len(original_files) == 0 or len(watermarked_files) == 0:
        print("Error: No audio files loaded. Please check the directory paths and ensure there are .wav files present.")
        return
    
    # Load and preprocess data in batches to avoid memory issues
    batch_size = 10  # Adjust the batch size as necessary
    target_sr = 44100  # Target sample rate for resampling
    max_length = 441000  # Example: 10 seconds at 44.1 kHz
    X_train, y_train = [], []
    
    for i in range(0, len(original_files), batch_size):
        original_batch_files = original_files[i:i + batch_size]
        watermarked_batch_files = watermarked_files[i:i + batch_size]
        
        original_batch = [load_audio_file(f, target_sr, max_length) for f in original_batch_files]
        watermarked_batch = [load_audio_file(f, target_sr, max_length) for f in watermarked_batch_files]
        
        X_batch, y_batch = preprocess_data(original_batch, watermarked_batch, max_length)
        
        if len(X_train) == 0:
            X_train, y_train = X_batch, y_batch
        else:
            X_train = np.concatenate((X_train, X_batch), axis=0)
            y_train = np.concatenate((y_train, y_batch), axis=0)
        
        with open(result_file_path, 'a') as result_file:
            result_file.write(f"Batch {i//batch_size + 1}:\n")
            result_file.write(f"  Original files: {original_batch_files}\n")
            result_file.write(f"  Watermarked files: {watermarked_batch_files}\n")
            result_file.write(f"  Loaded and preprocessed {len(original_batch)} files\n")
            result_file.write(f"  Current training data shape: X_train={X_train.shape}, y_train={y_train.shape}\n\n")
    
    # Debugging: Print the shape of the training data
    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    
    if X_train.size == 0 or y_train.size == 0:
        print("Error: Training data is empty. Ensure that the audio files are properly loaded and processed.")
        return
    
    # Build and train the model
    input_shape = (max_length, 1)
    autoencoder = build_autoencoder(input_shape)
    
    # Custom callback to log the training progress
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            with open(result_file_path, 'a') as result_file:
                result_file.write(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}\n")
    
    autoencoder.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2, callbacks=[CustomCallback()])
    
    # Save the model
    model_path = 'audio_watermark_env/audio_watermarking_model.h5'
    autoencoder.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Log the end time and model save path
    with open(result_file_path, 'a') as result_file:
        result_file.write(f"\nEnd Time: {datetime.now()}\n")
        result_file.write(f"Model saved to: {model_path}\n")

if __name__ == "__main__":
    main()
