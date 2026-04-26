import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

# 🔹 Load model safely (VERY IMPORTANT FIX)
model = tf.keras.models.load_model("heart_model.h5", compile=False)

# 🔹 Class labels
classes = ['AS', 'MS', 'MR', 'MVP', 'Normal']

# 🔹 UI
st.title("💓 Heart Valve Disorder Detection")
st.write("Upload a heart sound (.wav) file")

file = st.file_uploader("Upload PCG file", type=["wav"])

if file is not None:
    st.audio(file, format="audio/wav")

    try:
        # 🔹 Load audio
        signal, sr = librosa.load(file, sr=2000)

        # 🔹 Preprocessing
        max_len = 4000
        if len(signal) < max_len:
            signal = np.pad(signal, (0, max_len - len(signal)))
        else:
            signal = signal[:max_len]

        # 🔹 Spectrogram
        spec = librosa.feature.melspectrogram(
            y=signal,
            sr=2000,
            n_fft=512,
            hop_length=128,
            n_mels=128
        )

        spec_db = librosa.power_to_db(spec)
        spec_db = (spec_db - np.mean(spec_db)) / np.std(spec_db)

        # 🔹 Reshape
        spec_input = spec_db[np.newaxis, ..., np.newaxis]

        # 🔹 Predict
        prediction = model.predict(spec_input)
        predicted_class = classes[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        # 🔹 Output
        st.success(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")

        # 🔥 Show probabilities (NEW — looks professional)
        st.subheader("Prediction Probabilities")
        for i, cls in enumerate(classes):
            st.write(f"{cls}: {prediction[0][i]:.2f}")

        # 🔥 Spectrogram visualization
        st.subheader("Spectrogram")
        fig, ax = plt.subplots()
        img = librosa.display.specshow(spec_db, sr=2000, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")