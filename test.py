import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import numpy as np
import cv2
import mediapipe as mp
import joblib
from sklearn.preprocessing import LabelEncoder
import azure.cognitiveservices.speech as speechsdk
import time
from datetime import datetime
import av

# === Azure Speech Configuration ===
speech_key = "C2aQwIVI4DwKew11iZqZiOn4x1FEt7qgaM2qIfDZIdCXnZm9LEfMJQQJ99BEACYeBjFXJ3w3AAAEACOGfGDZ"
region = "eastus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
speech_config.speech_synthesis_voice_name = "sw-KE-ZuriNeural"
synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# === Load model and label encoder ===
model = joblib.load("model/mlp_tsl_static.pkl")
le = LabelEncoder()
le.fit([chr(i) for i in range(ord('A'), ord('Z') + 1)])

# === MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === Normalize landmarks ===
def normalize_landmarks(landmarks):
    coords = np.array(landmarks).reshape(-1, 3).astype(np.float32)
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    norm_coords = (coords - coords_min) / (coords_max - coords_min + 1e-6)
    return norm_coords.flatten().reshape(1, -1)

# === Streamlit UI State ===
if "word" not in st.session_state:
    st.session_state.word = ""
if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "last_letter" not in st.session_state:
    st.session_state.last_letter = ""
if "hold_start" not in st.session_state:
    st.session_state.hold_start = 0
if "last_seen" not in st.session_state:
    st.session_state.last_seen = time.time()

# === WebRTC Video Processor ===
class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_letter = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_time = time.time()
        current_letter = ""

        if results.multi_hand_landmarks:
            st.session_state.last_seen = current_time
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                try:
                    X = normalize_landmarks(landmarks)
                    pred_index = model.predict(X)[0]
                    current_letter = le.inverse_transform([pred_index])[0]

                    if current_letter == st.session_state.last_letter:
                        if st.session_state.hold_start == 0:
                            st.session_state.hold_start = current_time
                        if current_time - st.session_state.hold_start >= 1:
                            if not st.session_state.word or st.session_state.word[-1] != current_letter:
                                st.session_state.word += current_letter
                    else:
                        st.session_state.hold_start = current_time

                    st.session_state.last_letter = current_letter
                except Exception as e:
                    print("Prediction error:", e)
        else:
            delta = current_time - st.session_state.last_seen
            if delta >= 2 and st.session_state.word and not st.session_state.word.endswith(" "):
                st.session_state.word += " "
            if delta >= 5 and st.session_state.word.strip():
                st.session_state.sentence += st.session_state.word.strip() + " "
                st.session_state.word = ""

        self.result_letter = current_letter
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# === Streamlit Layout ===
st.set_page_config(page_title="BRIDGING SILENCE", layout="wide")
st.title("👐 BRIDGING SILENCE")

left_col, right_col = st.columns([3, 1])

with left_col:
    webrtc_streamer(
        key="sign-stream",
        video_processor_factory=SignProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
    )

    st.markdown(f"""
        ### Prediction:
        - **Letter**: `{st.session_state.last_letter}`
        - **Word**: `{st.session_state.word.strip()}`
        - **Sentence**: `{st.session_state.sentence.strip()}`
    """)

with right_col:
    if st.button("🗣 Speak"):
        full_text = (st.session_state.sentence + st.session_state.word).strip()
        if full_text:
            try:
                synthesizer.speak_text_async(full_text).get()
                st.success("Speaking: " + full_text)
                st.session_state.sentence = ""
                st.session_state.word = ""
            except Exception as e:
                st.error("Azure error: " + str(e))

    if st.button("❌ Clear"):
        st.session_state.word = ""
        st.session_state.sentence = ""
        st.session_state.last_letter = ""
        st.session_state.hold_start = 0

    if st.button("⌫ Delete Letter"):
        st.session_state.word = st.session_state.word[:-1]

    if st.button("🗑 Delete Word"):
        st.session_state.word = ""
