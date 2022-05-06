import gc
import os
import pathlib

import tensorflow
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
from PIL import ImageOps
import streamlit as st


current_dir = pathlib.Path.cwd()

st.set_page_config(
    page_title='Bengali Grapheme App',
    page_icon=':pencil:'
)

@st.cache(allow_output_mutation=True)
def load_data_and_model():
    model = tensorflow.keras.models.load_model(current_dir.joinpath('model', 'my_model'))
    class_map_df = pd.read_csv(current_dir.joinpath('input', 'class_map.csv'))
    return model, class_map_df

model, class_map_df = load_data_and_model()
sample_images_dir = current_dir.joinpath('input', 'sample-images')

def get_resized_image_roi(image, resize_size=64):
    image = image.reshape(image.shape[0], image.shape[1])
    _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

    idx = 0 
    ls_xmin = []
    ls_ymin = []
    ls_xmax = []
    ls_ymax = []
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        ls_xmin.append(x)
        ls_ymin.append(y)
        ls_xmax.append(x + w)
        ls_ymax.append(y + h)
    xmin = min(ls_xmin)
    ymin = min(ls_ymin)
    xmax = max(ls_xmax)
    ymax = max(ls_ymax)

    roi = image[ymin:ymax, xmin:xmax]
    resized_roi = cv2.resize(roi, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
    return resized_roi

preds_dict = {
    'grapheme_root': [],
    'vowel_diacritic': [],
    'consonant_diacritic': []
}

IMG_SIZE=64
N_CHANNELS=1

components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
target=[] # model predictions placeholder
row_id=[] # row_id place holder

st.title('Bengali Grapheme App')
st.header('Handwritten Character Recognition')
st.image(Image.open(current_dir.joinpath('assets', 'header.png')))
st.markdown('Background and dataset: [https://www.kaggle.com/c/bengaliai-cv19](https://www.kaggle.com/c/bengaliai-cv19)')

st.markdown(
    """ <style>
            div[role="radiogroup"] >  :first-child{
                display: none !important;
            }
        </style>
        """,
    unsafe_allow_html=True
)

def render_results(predictions):
    for i, p in enumerate(preds_dict):
        preds_dict[p] = np.argmax(predictions[i], axis=1)

    for k, id in enumerate(df_test_img.index.values):  
        for i, comp in enumerate(components):
            id_sample = id + '_' + comp
            row_id.append(id_sample)
            target.append(preds_dict[comp][k])

    df_predictions = pd.DataFrame(
        {
            'row_id': row_id,
            'target':target
        },
        columns = ['row_id','target'] 
    )

    test_images = df_test_img.values.reshape(-1, IMG_SIZE, IMG_SIZE)
        
    for i in range(int(df_predictions.shape[0]/3)):
        df_predictions_copy = df_predictions.copy()
        df_predictions_unit = df_predictions_copy.iloc[i*3: (i*3+3), :]
        df_predictions_unit['row_id'] = df_predictions_unit['row_id'].apply(lambda x: x.split('_', 2)[2])
        df_predictions_unit = df_predictions_unit.merge(class_map_df, 
                                                            how='inner', 
                                                            left_on=['row_id', 'target'], 
                                                            right_on=['component_type', 'label'])
        df_predictions_unit = df_predictions_unit.drop(columns=['row_id', 'label'])
        grapheme_list = []
        for j in df_predictions_unit['component']:
            grapheme_list.append(j)
        if grapheme_list[0] != "র্":
            temp = grapheme_list[0]
            grapheme_list[0] = grapheme_list[1]
            grapheme_list[1] = temp
        
        grapheme_list = list(filter(('0').__ne__, grapheme_list))
        grapheme = ''.join(grapheme_list)
        with st.container():
            with col1:
                st.image(test_images[i])
        with st.container():
            with col2:
                st.title(grapheme)

with st.form(key='my_form'):
    mode = st.radio(
        "Select input mode:",
        ('-', 'Use provided images', 'Upload own character images'))
    preprocessing = st.radio(
        "Select image mode:",
        ('-', 'Grayscale', 'Binary'))
    submit_button = st.form_submit_button(label='Submit')

col1, col2 = st.columns(2)

if mode == 'Use provided images':
    with st.container():
        with col1:
            st.header('Input Images:')
        with col2:
            st.header('Predictions:')
    
    resized = pd.DataFrame()
    resize_size = 64
    for i, image_path in enumerate(os.listdir(sample_images_dir)):
        image = cv2.imread(os.path.join(sample_images_dir, image_path), 0)
        resized_roi = get_resized_image_roi(image, resize_size)
        if preprocessing == "Binary":
            _, resized_roi = cv2.threshold(resized_roi, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized[f"image_{i}"] = resized_roi.reshape(-1)
    df_test_img = resized.T
    df_test_img = df_test_img/255

    X_test = df_test_img.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
    
    preds = model.predict(X_test)

    render_results(preds)
        
elif mode == 'Upload own character images':

    uploaded_files = st.file_uploader("Choose images to upload", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    
    resized = pd.DataFrame()
    resize_size = 64
    uploaded_file = None

    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        image = ImageOps.grayscale(image)
        image = np.array(image)
        resized_roi = get_resized_image_roi(image, resize_size)
        if preprocessing == "Binary":
            _, resized_roi = cv2.threshold(resized_roi, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized[f"image_{i}"] = resized_roi.reshape(-1)

    df_test_img = resized.T/255
    X_test = df_test_img.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    st.markdown('Samples available here ([**link**](https://drive.google.com/drive/folders/1rQzU2mUAuiZsGzpy3NqJgaFjSccFzYFv?usp=sharing)).')
    with st.container():
        with col1:
            st.header('Input Images:')
        with col2:
            st.header('Predictions:')

    if uploaded_file is not None:
        preds = model.predict(X_test)
        render_results(preds)

st.markdown('Hero image source: [https://www.kaggle.com/c/bengaliai-cv19](https://www.kaggle.com/c/bengaliai-cv19)')
