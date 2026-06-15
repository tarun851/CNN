import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

st.set_page_config(page_title="CNN - MNIST Classifier", layout="centered")
st.title("🖼️ Convolutional Neural Network (CNN)")
st.markdown("Classify handwritten digits (MNIST) using a CNN built with Keras.")


@st.cache_resource
def load_data_and_train():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train_n = X_train[:5000].reshape(-1, 28, 28, 1) / 255.0
    y_train_s = y_train[:5000]
    X_test_n = X_test[:1000].reshape(-1, 28, 28, 1) / 255.0
    y_test_s = y_test[:1000]

    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train_n, y_train_s, epochs=3, verbose=0)
    loss, acc = model.evaluate(X_test_n, y_test_s, verbose=0)
    return model, X_test_n, y_test_s, acc


with st.spinner("Training CNN on MNIST..."):
    model, X_test, y_test, acc = load_data_and_train()

# ----------------------------------------------------------------------
# 1. Dataset Preview
# ----------------------------------------------------------------------
st.header("📊 Dataset Preview")
st.markdown("A sample of handwritten digits from the MNIST test set:")

fig, axes = plt.subplots(1, 5, figsize=(8, 2))
for i, ax in enumerate(axes):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(str(y_test[i]), fontsize=10)
    ax.axis("off")
fig.tight_layout()
st.pyplot(fig)

# ----------------------------------------------------------------------
# 2. Model Training
# ----------------------------------------------------------------------
st.header("🏋️ Model Training")
st.success(f"CNN trained successfully! Test Accuracy: {acc * 100:.2f}%")

# ----------------------------------------------------------------------
# 3. Output - Old Data vs New Data Comparison
# ----------------------------------------------------------------------
st.header("🔮 Output")
st.markdown("Compare predictions on **two different test images** side by side.")

c1, c2 = st.columns(2)
with c1:
    old_idx = st.slider("Pick first image index (old data):", 0, len(X_test) - 1, 0)
with c2:
    new_idx = st.slider("Pick second image index (new data):", 0, len(X_test) - 1, 1)

if st.button("Compare Predictions"):
    old_pred = model.predict(X_test[old_idx:old_idx + 1], verbose=0)[0]
    new_pred = model.predict(X_test[new_idx:new_idx + 1], verbose=0)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 Old Data")
        fig1, ax1 = plt.subplots(figsize=(2.5, 2.5))
        ax1.imshow(X_test[old_idx].reshape(28, 28), cmap="gray")
        ax1.axis("off")
        st.pyplot(fig1)
        st.write(f"**Actual Label:** {y_test[old_idx]}")
        st.write(f"**Predicted Digit:** {np.argmax(old_pred)}")
        st.bar_chart(pd.DataFrame({"Probability": old_pred}, index=[str(i) for i in range(10)]))
    with col2:
        st.subheader("🆕 New Data")
        fig2, ax2 = plt.subplots(figsize=(2.5, 2.5))
        ax2.imshow(X_test[new_idx].reshape(28, 28), cmap="gray")
        ax2.axis("off")
        st.pyplot(fig2)
        st.write(f"**Actual Label:** {y_test[new_idx]}")
        st.write(f"**Predicted Digit:** {np.argmax(new_pred)}")
        st.bar_chart(pd.DataFrame({"Probability": new_pred}, index=[str(i) for i in range(10)]))