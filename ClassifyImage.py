# export CXXFLAGS=-I/opt/homebrew/include

from gooey import Gooey, GooeyParser
import logging 
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
import numpy as np
import os
#Modelo VGG16

logging.basicConfig(format='[%(asctime)s] - %(name)s - %(funcName)s - %(levelname)s : %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

def read_image(file_path):
    log.info("Loading and preprocessing image...")  
    image = load_img(file_path, target_size=(224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)
    image /= 255.  
    return image

def test_single_image(path, vgg16, model):
    animals = ['falso', 'mamifero', 'pajarochico']
    images = read_image(path)
    bt_prediction = vgg16.predict(images)  
    preds = model.predict(bt_prediction)
    print(12*"-"+"Classifier results"+12*"-")
    for idx, animal, x in zip(range(3), animals, preds[0]):
        print("ID: {}, Label: {} {}%".format(idx, animal, round(x*100,2) ))
 
      
@Gooey
def main():
    parser = GooeyParser(description="My Cool GUI Program!") 
    parser.add_argument('Image', widget="FileChooser")

    args = parser.parse_args()
    args_dict = vars(args)

    vgg16 = load_model("vgg16.h5", compile=False)
    model = load_model("model1.h5", compile=False)

    test_single_image(args_dict["Image"], vgg16, model)

if __name__ == "__main__":
    main()