import streamlit as st
import tensorflow as tf


@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('E:/project/snake_vision/snake_clf_inceptionv3.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # Snake Classification
         """
         )
st.markdown('This app can classify images of snakes from 5 different species given below ')

st.markdown(['Black Rat snake','Common Garter snake','DeKay Brown snake','Northern Watersnake','Western Diamondback rattlesnake'])

file = st.file_uploader("Please upload an image", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        return prediction
    
if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Black_Rat_snake','Common_Garter_snake','DeKay_Brown_snake','Northern_Watersnake','Western_Diamondback_rattlesnake']
    string = 'This image is '+class_names[np.argmax(predictions)]
    st.success(string)
