import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

st.write('''
# Identifikasi Jamur Akar Putih Pada Pohon Karet
''')
st.write('''Sebuah Web App Klasifikasi Citra yang Dapat Mengidentifikasi Jamur Akar Putih Pada Pohon Karet''')

file = st.file_uploader("Silahkan masukkan gambar pohon", type=['jpg','png'])


def predict_stage(image_data,model):
    size = (224, 224)
    image = ImageOps.fit(image_data,size, Image.ANTIALIAS)
    image_array = np.array(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    preds = ""
    prediction = model.predict(data)
    if prediction<=0.5:
        st.write(f"Berjamur")
    elif prediction>=0.5:
        st.write(f"Pohon Sehat")
   # else:
    #    st.write(f"Terimakasih")
    return prediction

if file is None:
    st.text("Silahkan masukkan file gambar")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    model = tf.keras.models.load_model('modelJAPbaru.h5')
    Generate_pred = st.button("Deteksi Jamur...")
    if Generate_pred:
        prediction = predict_stage(image, model)
        # st.text("Probabilitas (0: Berjamur, 1: Sehat)")
        # st.write(prediction)




    