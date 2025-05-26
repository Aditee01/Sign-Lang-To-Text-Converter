# Sign-Lang-To-Text-Converter
A real-time American Sign Language (ASL) recognition system that converts hand gestures into text using a CNN model. The system features a user-friendly GUI built with Tkinter and OpenCV for camera capture.

Features
✅ Real-time ASL alphabet detection (A-Z, excluding J/Z)
✅ CNN model trained on 87,000+ ASL images
✅ Works with webcam or video input

Clone the repository:

git clone https://github.com/yourusername/Sign-Lang-To-Text-Converter
cd Sign-Lang-To-Text-Converter
Install dependencies:


pip install -r requirements.txt

1. Real-time Detection (Webcam)

python app.py

2. Train the CNN Model

Dataset
Download the ASL Alphabet Dataset from Kaggle (~87,000 images).

Project Structure
sign-language-converter/
├── asl_alphabet_train/          # Dataset directory

│   └── asl_alphabet_train/

│       ├── A/                   # Sample images for letter A

│       ├── B/

│       └── ...                  # All 29 classes

├── asl-alphabet-dataset.py      # Model training script

├── app.py                       # Main application

├── requirements.txt             # Dependency list

├── asl_model.h5                 # Trained model (generated)

|-- label.npy

└── README.md   
