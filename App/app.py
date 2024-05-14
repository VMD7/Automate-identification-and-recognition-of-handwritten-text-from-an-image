import gradio as gr
from keras.models import load_model
import keras.backend as K
import cv2
import numpy as np

try:
  model = load_model('Text_recognizer_Using_CRNN.h5')
except Exception as e:
  print("Not able to load the trained model.")

def process_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    w, h = img.shape
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = np.expand_dims(img, axis=2)

    # Normalize
    img = img / 255

    return img

def predict_image_text(input_img):
  try:
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    img = process_image(img)
    prediction =model.predict(np.asarray([img]))
    decoded = K.ctc_decode(prediction,
                          input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                          greedy=True)[0][0]
    out = K.get_value(decoded)
    char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    # see the results
    temp_text=""
    for i, x in enumerate(out):
        for p in x:
            if int(p) != -1:
                temp_text+=char_list[int(p)]
    return temp_text
  except:
    return "Some error occured please try again with proper image."

import gradio as gr

demo = gr.Interface(predict_image_text, gr.Image(), "text",title="Image to Text Conversion",description="Upload an image and get the extracted text.")
demo.launch()
