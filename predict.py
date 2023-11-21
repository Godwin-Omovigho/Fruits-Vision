import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
import numpy as np
from PIL import Image


train_dir ='Train'


data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))

loaded_model = tf.keras.models.load_model('Fruit Model')


# Create a function to import an image and resize it to be able to be used with our model
def load_and_pred_image(filename,img_shape=224):
  
  #  Load the image
  image = Image.open(filename)

  image = np.array(image)

  #Resize the image
  image=tf.image.resize(image, size=[img_shape,img_shape])

  #Rescale the image
  image = image/255
  return image


def pred_and_plot(model, filename,class_names=class_names):
  """
  Imports an image located at filename, makes a prediction with model and plots the
  image with the predicted class as the title"""

  confidence_threshold=0.45

  #Import the target image and preprocess it
  img=load_and_pred_image(filename)

  #Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  #Get the predicted class
  pred_class = np.argmax(pred)
  # pred_class=class_names[pred.argmax()]

  confidence = pred[0][pred_class]

  # Check if the confidence is above the threshold
  if confidence >= confidence_threshold:
    class_name = class_names[pred_class]
    return class_name
    
  else:
    return "Sorry, this doesn't appear to be a fruit. Please try another image."

  # return pred_class


def show_predict_page():

    # Custom CSS for styling
    custom_css = """
    <style>

      .title {
              text-align: center;
              color: white;
              background-color: #3F51B5; /* Background color is blue */
              padding: 10px; /* Add some padding for spacing */
          }
        .header {
            color: white;
            background-color: #3F51B5; /* Background color is blue */
            padding: 5px 5px; /* Add some padding for spacing */
        }
        .stText {
            color: black; /* Change text color to black */
        }
        .header-text {
            color: white; /* Change text color to white for specific headers */
        }
    </style>
    """
    
    # Apply the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Streamlit app
    st.markdown("<h1 class='title'>Fruit Recognition App</h1>", unsafe_allow_html=True)
    
    # List of fruits
    fruits = ['apple', 'banana', 'cherry', 'chickoo', 'grapes', 'kiwi', 'mango', 'orange', 'strawberry']

    # Description of your computer vision model
    model_description = "This computer vision model is adept at recognizing and categorizing nine different types of fruits. It has been trained on a diverse dataset consisting of 294 training images and 72 test images.."
    
    
    st.markdown("\n\n\n")
    st.markdown("<h2 class='header'> Model Description</h2>", unsafe_allow_html=True)
    st.write(model_description)

    st.markdown("<h2 class='header'>List of Fruits</h2>", unsafe_allow_html=True)
    
    
    st.write(f"Here are the fruits the model was trained on:")
    
    # Display the indexed list of fruits starting from 1
    for i, fruit in enumerate(fruits, start=1):
      st.write(f"{i}. {fruit}")


    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
    # Get the image path
      image_path = st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

     # Make prediction using the prediction function
      predicted_class_label = pred_and_plot(loaded_model,uploaded_image,class_names)

    # Display the prediction
      st.write('Predicted class label:', predicted_class_label)
        
