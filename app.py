import numpy as np
import mediapipe as mp
import joblib
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from PIL import Image, ImageTk
import os
import time
from datetime import datetime
import cv2
import azure.cognitiveservices.speech as speechsdk

# === Azure Speech API credentials ===
speech_key = "C2aQwIVI4DwKew11iZqZiOn4x1FEt7qgaM2qIfDZIdCXnZm9LEfMJQQJ99BEACYeBjFXJ3w3AAAEACOGfGDZ"
service_region = "eastus"

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_synthesis_voice_name = "sw-KE-ZuriNeural"
synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# === Load model and labels ===
model = joblib.load('model/mlp_tsl_static.pkl')
le = LabelEncoder()
le.fit([chr(i) for i in range(ord('A'), ord('Z') + 1)])

# === MediaPipe setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# === Normalize landmarks ===
def normalize_landmarks(landmarks):
    coords = np.array(landmarks).reshape(-1, 3).astype(np.float32)
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    norm_coords = (coords - coords_min) / (coords_max - coords_min + 1e-6)
    return norm_coords.flatten().reshape(1, -1)

# === GUI Application ===
class TSLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSL - Bridging Silence")
        self.root.configure(bg="#f0f4f8")

        self.video_running = False
        self.cap = None

        self.prev_letter = ""
        self.letter_hold_start = None
        self.last_seen_time = time.time()
        self.word = ""
        self.sentence = ""
        self.saved_sentences = []

        # === Title (Centered) ===
        self.title_label = tk.Label(root, text="BRIDGING SILENCE", font=("Arial", 24, "bold"),
                                    fg="#005073", bg="#f0f4f8")
        self.title_label.pack(pady=10)

        # === Horizontal Layout Container ===
        self.container = tk.Frame(root, bg="#f0f4f8")
        self.container.pack(fill="both", expand=True)

        # === Left Panel: Video and Text ===
        self.left_panel = tk.Frame(self.container, bg="#f0f4f8")
        self.left_panel.pack(side="left", padx=10, pady=10)

        self.video_label = tk.Label(self.left_panel, bg="#e6ecf0", bd=2, relief="solid")
        self.video_label.pack()

        self.prediction_label = tk.Label(
            self.left_panel,
            text="Letter: \nWord: \nSentence:",
            font=("Arial", 16),
            fg="#005073",
            bg="#f0f4f8",
            justify="left",
            anchor="w",
            padx=10,
            pady=10
        )
        self.prediction_label.pack(pady=10, anchor="w")

        # === Right Panel: Control Buttons ===
        self.controls = tk.Frame(self.container, bg="#f0f4f8")
        self.controls.pack(side="right", padx=20, pady=10, fill="y")

        button_style = {"font": ("Arial", 12), "width": 12, "padx": 5, "pady": 5}

        tk.Button(self.controls, text="Start", command=self.start_video,
                  bg="#28a745", fg="white", **button_style).pack(pady=4)
        tk.Button(self.controls, text="Stop", command=self.stop_video,
                  bg="#dc3545", fg="white", **button_style).pack(pady=4)
        tk.Button(self.controls, text="Clear", command=self.clear_predictions,
                  bg="#ffc107", **button_style).pack(pady=4)
        tk.Button(self.controls, text="Speak", command=self.speak_text,
                  bg="#17a2b8", fg="white", **button_style).pack(pady=4)
        tk.Button(self.controls, text="Del Letter", command=self.delete_last_letter,
                  bg="#6c757d", fg="white", **button_style).pack(pady=4)
        tk.Button(self.controls, text="Del Word", command=self.delete_last_word,
                  bg="#343a40", fg="white", **button_style).pack(pady=4)

    def start_video(self):
        if not self.video_running:
            self.cap = cv2.VideoCapture(0)
            self.video_running = True
            self.update_video()

    def stop_video(self):
        self.video_running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')

    def clear_predictions(self):
        self.word = ""
        self.sentence = ""
        self.saved_sentences.clear()
        self.prediction_label.config(text="Letter: \nWord: \nSentence:")

    def speak_text(self):
        full_sentence = (self.sentence + self.word).strip()
        if full_sentence:
            try:
                result = synthesizer.speak_text_async(full_sentence).get()
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    print("Speech synthesis completed.")
                else:
                    print("Speech synthesis failed:", result.reason)
            except Exception as e:
                print("Azure speech error:", e)

            self.saved_sentences.append(full_sentence)
            self.word = ""
            self.sentence = ""
            self.prediction_label.config(text="Letter: \nWord: \nSentence:")

    def delete_last_letter(self):
        if self.word:
            self.word = self.word[:-1]
        elif self.sentence:
            self.sentence = self.sentence.rstrip()
            if self.sentence and self.sentence[-1] == " ":
                self.sentence = self.sentence[:-1]
            self.word = self.sentence.split()[-1] if self.sentence else ""
            self.sentence = " ".join(self.sentence.split()[:-1]) + " "
        self.prediction_label.config(text=f"Letter: \nWord: {self.word}\nSentence: {self.sentence}")

    def delete_last_word(self):
        if self.word:
            self.word = ""
        elif self.sentence:
            self.sentence = self.sentence.rstrip()
            words = self.sentence.split()
            self.sentence = " ".join(words[:-1]) + " " if words else ""
        self.prediction_label.config(text=f"Letter: \nWord: {self.word}\nSentence: {self.sentence}")

    def update_video(self):
        if not self.video_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_time = time.time()
        current_letter = ""

        if results.multi_hand_landmarks:
            self.last_seen_time = current_time
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                try:
                    X = normalize_landmarks(landmarks)
                    pred_index = model.predict(X)[0]
                    current_letter = le.inverse_transform([pred_index])[0]

                    if current_letter == self.prev_letter:
                        if not self.letter_hold_start:
                            self.letter_hold_start = current_time
                        if current_time - self.letter_hold_start >= 1:
                            if not self.word or self.word[-1] != current_letter:
                                self.word += current_letter
                    else:
                        self.letter_hold_start = current_time

                    self.prev_letter = current_letter

                except Exception as e:
                    print("Prediction error:", e)
        else:
            # When no hand is detected
            time_since_last = current_time - self.last_seen_time
            if time_since_last >= 2 and self.word and (not self.word.endswith(" ")):
                self.word += " "
            if time_since_last >= 5 and self.word.strip():
                self.sentence += self.word.strip() + " "
                self.word = ""

        display_text = f"Letter: {current_letter}\nWord: {self.word}\nSentence: {self.sentence}"
        self.prediction_label.config(text=display_text)

        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)

# === Launch the GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    app = TSLApp(root)
    root.mainloop()
