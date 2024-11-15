import streamlit as st
import cv2
import os
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

model = load_model('FV.h5')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']


# Define a function to preprocess the image
def preprocess_image(image_path):
    img2 = cv2.imread(image_path)

    # Resize the image to 224x224
    img  = cv2.resize(img2, (224, 224))

    preprocess_folder = 'preprocess'
    os.makedirs(preprocess_folder, exist_ok=True)
    
    # Step 1: Denoising
    denoised_image = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    denoised_path = os.path.join(preprocess_folder, 'denoised.jpg')
    cv2.imwrite(denoised_path, denoised_image)
    
    # Step 2: Segmentation (using simple threshold for demonstration)
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    _, segmented_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    segmented_path = os.path.join(preprocess_folder, 'segmented.jpg')
    cv2.imwrite(segmented_path, segmented_image)

    # Step 3: SIFT (detecting and drawing keypoints)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    sift_image = cv2.drawKeypoints(denoised_image, keypoints, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    sift_path = os.path.join(preprocess_folder, 'sift.jpg')
    cv2.imwrite(sift_path, sift_image)

    return denoised_path, segmented_path, sift_path


def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't fetch the Calories")
        print(e)


def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(" ".join(str(x) for x in y_class))
    res = labels[y]
    return res.capitalize()


def run():
    st.title("Fruits🍍-Vegetable🍅 Classification using transfer learning")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)

        # Save the uploaded image
        save_dir = './upload_images/'
        os.makedirs(save_dir, exist_ok=True)
        save_image_path = os.path.join(save_dir, img_file.name)
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Preprocess the image
        denoised_image_path, segmented_image_path, surf_image_path = preprocess_image(save_image_path)
        
        # Display preprocessed images
        st.image(denoised_image_path, caption="Denoised Image")
        st.image(segmented_image_path, caption="Segmented Image")
        st.image(surf_image_path, caption="sift Keypoints Image")
        
        # Classification on the denoised image
        result = processed_img(denoised_image_path)
        
        if result in vegetables:
            st.info('**Category : Vegetables**')
        else:
            st.info('**Category : Fruit**')
        
        st.success("**Predicted : " + result + '**')
        
        # Fetch and display calorie information
        cal = fetch_calories(result)
        if cal:
            st.warning('**' + cal + ' (100 grams)**')


run()
