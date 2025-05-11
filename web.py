import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import tempfile

model = load_model("cnn.h5")

def preprocess_image(img):
    img = img.convert("L")  
    img = img.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    norm = resized.astype("float32") / 255.0
    return norm.reshape(1, 28, 28, 1)

st.title("Real-Time Digit Classifier")
st.write("Upload an image or use your webcam to classify handwritten digits (MNIST).")

tab1, tab2 = st.tabs(["Upload Image", "Webcam"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)
        if st.button("Predict from Image"):
            input_img = preprocess_image(image)
            prediction = model.predict(input_img)
            predicted = np.argmax(prediction)
            st.success(f"Prediction: **{predicted}** with {np.max(prediction):.2%} confidence")

with tab2:
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])
    cap = None

    if run:
        cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not detected.")
            break

        frame_display = frame.copy()
    
        x, y, w = 100, 100, 200
        cv2.rectangle(frame_display, (x, y), (x + w, y + w), (0, 255, 0), 2)

        roi = frame[y:y + w, x:x + w]
        try:
            processed = preprocess_frame(roi)
            prediction = model.predict(processed)
            label = np.argmax(prediction)
            confidence = np.max(prediction)
            cv2.putText(frame_display, f"{"Predicition"} {label} ({confidence:.2%})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        except Exception as e:
            cv2.putText(frame_display, "Error processing", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))

    if cap:
        cap.release()
