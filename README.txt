____Directory Structure___

audio_watermark_env/
├── data/
│   ├── original/         				# Directory for original audio files
│   ├── watermarked/     			    # Directory for watermarked audio files
│   ├── new_audio/       			    # Directory for new audio files to be watermarked
│   ├── results/          				# Directory for saving watermarked results
│   ├── result_pics/      				# Directory for saving result plots
│   └── watermarking_utilization.txt   # Log file for watermarking process
└── audio_watermarking_model.h5        # Trained model file

_____________________________INSTRUCTIONS____________________________________________________________

_____Setup_____

1, Create a virtual environment:
Command: 
python -m venv audio_watermark_env
2, Activate the Virtual Environment:
Command:
audio_watermark_env\Scripts\activate
3, Install Dependencies:
Command:
pip install numpy librosa soundfile scipy tensorflow keras matplotlib
4, Select the Correct Interpreter in Your IDE
If you are using an IDE such as Visual Studio Code (VS Code), you can select the virtual environment as your Python interpreter.
+ Open VS Code.
+ Open the Command Palette (F1 or Ctrl+Shift+P).
+ Search for and select "Python: Select Interpreter".
+ Choose the interpreter located in your virtual environment:
	+ On Windows: .\audio_watermark_env\Scripts\python.exe
	+ On macOS/Linux: ./audio_watermark_env/bin/python


_____Usage_____

1, Generate Watermarked Audio
Run the generate_watermarked_audio.py script to embed watermarks into original audio files.
Command:
python generate_watermarked_audio.py
2, Train the Model
Run the model_training.py script to train the machine learning model using the original audio files.
Command:
python model_training.py
3, Analyze Original Audio
Run the original_audio_analysis.py script to analyze and extract features from the original audio files.
Command:
python original_audio_analysis.py
4, Utilize Watermarking
Run the watermarking_utilization.py script to embed watermarks into new audio files using the trained model.
Command:
python watermarking_utilization.py

