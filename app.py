import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import time
from collections import deque


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Sign Language to Text Converter")

        # Load the improved model and class names
        try:
            self.model = load_model('asl_model.h5')
            self.label = np.load('label.npy', allow_pickle=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model files: {str(e)}")
            self.root.destroy()
            return

        # Variables
        self.camera_on = False
        self.cap = None
        self.current_text = ""
        self.sentence = ""
        self.last_prediction_time = 0
        self.prediction_interval = 1.5  # Slightly faster prediction interval
        self.prediction_history = deque(maxlen=5)  # Track last 5 predictions
        self.confidence_threshold = 0.85  # Higher confidence threshold
        self.current_confidence = 0

        # Word buffer for sentence formation
        self.word_buffer = ""
        self.last_word_time = time.time()
        self.word_delay = 2.0  # Seconds before adding to sentence

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Configure main window
        self.root.geometry("900x750")
        self.root.minsize(800, 700)

        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Camera frame
        self.camera_frame = ttk.LabelFrame(self.main_frame, text="Sign Detection Area")
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_label = ttk.Label(self.camera_frame)
        self.video_label.pack(padx=5, pady=5)

        # Text display frame
        self.text_frame = ttk.LabelFrame(self.main_frame, text="Translation Output")
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Current detection frame
        self.detection_frame = ttk.Frame(self.text_frame)
        self.detection_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.detection_frame, text="Current Sign:", font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT)
        self.current_label = ttk.Label(self.detection_frame, text="None", font=('Helvetica', 12))
        self.current_label.pack(side=tk.LEFT, padx=5)

        ttk.Label(self.detection_frame, text="Confidence:", font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT,
                                                                                                 padx=(20, 5))
        self.confidence_label = ttk.Label(self.detection_frame, text="0%", font=('Helvetica', 12))
        self.confidence_label.pack(side=tk.LEFT)

        # Word buffer display
        self.buffer_frame = ttk.Frame(self.text_frame)
        self.buffer_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.buffer_frame, text="Word Buffer:", font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT)
        self.buffer_label = ttk.Label(self.buffer_frame, text="", font=('Helvetica', 12))
        self.buffer_label.pack(side=tk.LEFT, padx=5)

        # Sentence display
        self.sentence_frame = ttk.Frame(self.text_frame)
        self.sentence_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.sentence_frame, text="Sentence:", font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT)
        self.sentence_label = ttk.Label(self.sentence_frame, text="", font=('Helvetica', 12))
        self.sentence_label.pack(side=tk.LEFT, padx=5)

        # Text output with scrollbar
        self.output_frame = ttk.Frame(self.text_frame)
        self.output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.text_scroll = ttk.Scrollbar(self.output_frame)
        self.text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_output = tk.Text(
            self.output_frame,
            height=8,
            width=60,
            font=('Helvetica', 12),
            wrap=tk.WORD,
            yscrollcommand=self.text_scroll.set
        )
        self.text_output.pack(fill=tk.BOTH, expand=True)
        self.text_scroll.config(command=self.text_output.yview)

        # Control buttons
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        button_config = {
            'padding': (10, 5),
            'width': 12
        }

        self.start_btn = ttk.Button(
            self.control_frame,
            text="Start Camera",
            command=self.start_camera,
            **button_config
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            self.control_frame,
            text="Stop Camera",
            command=self.stop_camera,
            state=tk.DISABLED,
            **button_config
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.add_space_btn = ttk.Button(
            self.control_frame,
            text="Add Space",
            command=self.add_space,
            **button_config
        )
        self.add_space_btn.pack(side=tk.LEFT, padx=5)

        self.del_btn = ttk.Button(
            self.control_frame,
            text="Delete Last",
            command=self.delete_last,
            **button_config
        )
        self.del_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(
            self.control_frame,
            text="Clear All",
            command=self.clear_all,
            **button_config
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_bar = ttk.Label(
            self.main_frame,
            text="Ready - Click 'Start Camera' to begin",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 5)
        )
        self.status_bar.pack(fill=tk.X, padx=5, pady=5)

    def start_camera(self):
        if not self.camera_on:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return

            self.camera_on = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_bar.config(text="Camera active - Position your hand in the green box")
            self.update_camera()

    def stop_camera(self):
        if self.camera_on:
            self.camera_on = False
            self.cap.release()
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.video_label.config(image=None)
            self.status_bar.config(text="Camera stopped")

    def update_camera(self):
        if self.camera_on:
            ret, frame = self.cap.read()
            if ret:
                # Process frame for display
                frame = cv2.flip(frame, 1)

                # Get frame dimensions
                height, width = frame.shape[:2]

                # Define ROI (Region of Interest) - centered green box
                box_size = 250  # Slightly larger detection area
                box_x1 = width // 2 - box_size // 2
                box_y1 = height // 2 - box_size // 2
                box_x2 = box_x1 + box_size
                box_y2 = box_y1 + box_size

                # Draw green rectangle around ROI
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)

                # Add instruction text
                cv2.putText(
                    frame,
                    "Place hand here",
                    (box_x1, box_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Extract ROI for prediction
                roi = frame[box_y1:box_y2, box_x1:box_x2]

                # Make prediction every interval
                current_time = time.time()
                if current_time - self.last_prediction_time > self.prediction_interval:
                    prediction, confidence = self.predict_sign(roi)

                    if prediction is not None and confidence >= self.confidence_threshold:
                        self.current_text = prediction
                        self.current_confidence = confidence
                        self.current_label.config(text=prediction.upper())
                        self.confidence_label.config(text=f"{confidence:.0f}%")

                        # Add to prediction history
                        self.prediction_history.append(prediction)

                        # Update word buffer based on prediction
                        self.update_word_buffer(prediction)

                        self.last_prediction_time = current_time

                # Check if we should add word to sentence
                self.check_word_buffer()

                # Convert to PhotoImage
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(10, self.update_camera)

    def predict_sign(self, image):
        try:
            # Preprocess the image
            processed_img = cv2.resize(image, (224, 224))  # Match model input size
            processed_img = processed_img / 255.0
            processed_img = np.expand_dims(processed_img, axis=0)

            # Make prediction
            prediction = self.model.predict(processed_img)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction)

            predicted_class = self.label[class_idx]
            return predicted_class, confidence

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0

    def update_word_buffer(self, prediction):
        current_time = time.time()

        # Handle special commands
        if prediction == 'del':
            self.word_buffer = self.word_buffer[:-1]
        elif prediction == 'space':
            self.add_word_to_sentence()
            self.sentence += ' '
            self.update_sentence_display()
        elif prediction == 'nothing':
            return
        else:
            # Only add letter if we have a consistent prediction
            if len(self.prediction_history) == self.prediction_history.maxlen:
                # Check if all predictions are the same
                if len(set(self.prediction_history)) == 1:
                    self.word_buffer += prediction.lower()
                    self.last_word_time = current_time

        self.buffer_label.config(text=self.word_buffer)

    def check_word_buffer(self):
        current_time = time.time()
        if (self.word_buffer and
                (current_time - self.last_word_time) > self.word_delay):
            self.add_word_to_sentence()

    def add_word_to_sentence(self):
        if self.word_buffer:
            self.sentence += self.word_buffer
            self.word_buffer = ""
            self.update_sentence_display()
            self.buffer_label.config(text="")

    def update_sentence_display(self):
        self.sentence_label.config(text=self.sentence)
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, self.sentence)
        self.text_output.see(tk.END)

    def add_space(self):
        self.sentence += ' '
        self.update_sentence_display()

    def delete_last(self):
        self.sentence = self.sentence[:-1]
        self.update_sentence_display()

    def clear_all(self):
        self.sentence = ""
        self.word_buffer = ""
        self.current_text = ""
        self.current_confidence = 0
        self.current_label.config(text="None")
        self.confidence_label.config(text="0%")
        self.sentence_label.config(text="")
        self.buffer_label.config(text="")
        self.text_output.delete(1.0, tk.END)
        self.prediction_history.clear()

    def on_closing(self):
        if self.camera_on:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()