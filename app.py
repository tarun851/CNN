import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ------------------------------------------------

st.set_page_config(
    page_title="CNN CIFAR10",
    layout="wide"
)

os.makedirs(
    "models",
    exist_ok=True
)

# ------------------------------------------------

classes = [

"airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck"

]

# ------------------------------------------------
# SESSION
# ------------------------------------------------

defaults = {

"model":None,
"history":None,
"trained":False

}

for k,v in defaults.items():

    if k not in st.session_state:
        st.session_state[k]=v


# ------------------------------------------------
# DATASET
# ------------------------------------------------

@st.cache_data

def load_dataset():

    (
        X_train,
        y_train
    ),(
        X_test,
        y_test
    ) = tf.keras.datasets.cifar10.load_data()

    X_train = X_train/255.0
    X_test = X_test/255.0

    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


(
X_train,
y_train,
X_test,
y_test

)=load_dataset()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title(
    "CNN Settings"
)

epochs = st.sidebar.slider(
    "Epochs",
    5,
    30,
    10
)

batch = st.sidebar.slider(
    "Batch",
    16,
    128,
    32
)

lr = st.sidebar.selectbox(
    "Learning Rate",
    [
        0.001,
        0.0005,
        0.0001
    ]
)

train = st.sidebar.button(
    "Train Model"
)

# ------------------------------------------------

tab1,tab2,tab3,tab4,tab5 = st.tabs([

"Dataset",
"Training",
"Visualizations",
"Prediction",
"Summary"

])

# ------------------------------------------------
# DATASET TAB
# ------------------------------------------------

with tab1:

    st.header(
        "Dataset Preview"
    )

    st.success(
        "CIFAR-10 downloaded automatically"
    )

    fig = plt.figure(
        figsize=(10,6)
    )

    for i in range(12):

        plt.subplot(
            3,
            4,
            i+1
        )

        plt.imshow(
            X_train[i]
        )

        plt.title(
            classes[
                y_train[i][0]
            ]
        )

        plt.axis(
            "off"
        )

    st.pyplot(fig)

# ------------------------------------------------
# TRAIN
# ------------------------------------------------

with tab2:

    st.header(
        "Training"
    )

    if train:

        model = tf.keras.Sequential([

            tf.keras.layers.Conv2D(
                32,
                3,
                activation="relu",
                input_shape=(32,32,3)
            ),

            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(
                64,
                3,
                activation="relu"
            ),

            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(
                128,
                3,
                activation="relu"
            ),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(
                256,
                activation="relu"
            ),

            tf.keras.layers.Dropout(
                0.3
            ),

            tf.keras.layers.Dense(
                10,
                activation="softmax"
            )
        ])

        model.compile(

            optimizer=tf.keras.optimizers.Adam(
                lr
            ),

            loss="sparse_categorical_crossentropy",

            metrics=[
                "accuracy"
            ]
        )

        progress = st.progress(
            0
        )

        class Callback(
            tf.keras.callbacks.Callback
        ):

            def on_epoch_end(
                self,
                epoch,
                logs=None
            ):

                progress.progress(
                    (
                        epoch+1
                    )
                    /epochs
                )

        history = model.fit(

            X_train,
            y_train,

            validation_split=0.2,

            epochs=epochs,

            batch_size=batch,

            callbacks=[
                Callback()
            ]
        )

        model.save(
            "models/cnn.keras"
        )

        st.session_state.model=model
        st.session_state.history=history
        st.session_state.trained=True

        st.success(
            "Training Finished"
        )

# ------------------------------------------------
# VISUALIZE
# ------------------------------------------------

with tab3:

    if st.session_state.history:

        h = (
            st.session_state
            .history
            .history
        )

        st.subheader(
            "Loss"
        )

        fig=plt.figure()

        plt.plot(
            h["loss"]
        )

        plt.plot(
            h["val_loss"]
        )

        plt.legend([
            "Train",
            "Validation"
        ])

        st.pyplot(fig)

        st.subheader(
            "Accuracy"
        )

        fig=plt.figure()

        plt.plot(
            h["accuracy"]
        )

        plt.plot(
            h["val_accuracy"]
        )

        st.pyplot(fig)

        pred = np.argmax(

            st.session_state
            .model
            .predict(
                X_test
            ),

            axis=1

        )

        cm = confusion_matrix(

            y_test,
            pred

        )

        fig=plt.figure(
            figsize=(10,8)
        )

        sns.heatmap(
            cm,
            cmap="Blues"
        )

        st.pyplot(fig)

        st.text(

            classification_report(

                y_test,
                pred

            )

        )

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

with tab4:

    if st.session_state.trained:

        img = st.file_uploader(

            "Upload Image",

            [
                "jpg",
                "png",
                "jpeg"
            ]

        )

        if img:

            image = Image.open(
                img
            )

            st.image(
                image
            )

            image=image.resize(
                (
                    32,
                    32
                )
            )

            x=np.array(
                image
            )/255

            x=np.expand_dims(
                x,
                0
            )

            pred = (

                st.session_state
                .model
                .predict(
                    x
                )

            )

            c=np.argmax(
                pred
            )

            st.success(

                classes[
                    c
                ]

            )

            st.progress(

                float(
                    pred[0][c]
                )

            )

# ------------------------------------------------
# SUMMARY
# ------------------------------------------------

with tab5:

    if st.session_state.trained:

        st.metric(

            "Final Accuracy",

            round(

                st.session_state
                .history
                .history[
                    "val_accuracy"
                ][-1]

                *100,

                2

            )

        )

        st.metric(
            "Epochs",
            epochs
        )

        st.write(
            st.session_state
            .model
            .summary()
        )

    else:

        st.info(
            "Train model first"
        )