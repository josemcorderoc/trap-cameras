# export CXXFLAGS=-I/opt/homebrew/include

from gooey import Gooey, GooeyParser
import logging 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img 
import numpy as np
import os
import shutil

logging.basicConfig(format='[%(asctime)s] - %(name)s - %(funcName)s - %(levelname)s : %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

animals = ['falso', 'mamifero', 'pajarochico']

def read_image(file_path: str):
    log.debug("Loading and preprocessing image...")  
    image = load_img(file_path, target_size=(224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)
    image /= 255.  
    return image

def classify_multiple_images(images_paths, vgg16, model):
    images = np.vstack([read_image(img_path) for img_path in images_paths])
    
    bt_prediction = vgg16.predict(images)  
    preds = model.predict(bt_prediction)
    
    # el axis elegido es el que muere
    y_classes = np.argmax(preds, axis=1) 
    predicted_labels = np.array(animals)[y_classes]

    return dict(zip(images_paths, predicted_labels))

@Gooey(program_name="Clasificador de imágenes por carpeta", default_size=(1000, 400))
def main():
    parser = GooeyParser(description=f"Este programa recibe una carpeta con imágenes y entrega los resultados clasificados en las categorías {animals}") 
    
    parser.add_argument('Carpeta de imágenes', widget="DirChooser", help="La carpeta con las imágenes a clasificar")
    parser.add_argument('Carpeta de salida', widget="DirChooser", help="Debe seleccionar una carpeta vacía, en ella se escribirán las imágenes clasificadas")

    args = parser.parse_args()
    args_dict = vars(args)
    img_dir_path = args_dict["Carpeta de imágenes"]
    output_dir_path = args_dict["Carpeta de salida"]

    vgg16 = load_model("vgg16.h5", compile=False)
    model = load_model("model1.h5", compile=False)

    images_paths = [os.path.join(img_dir_path, img) for img in os.listdir(img_dir_path) if not img.startswith(".")]
    log.info(f"Se encontraron {len(images_paths)} archivos")
    
    log.info(f"Empezando predicciones...")
    predictions = classify_multiple_images(images_paths, vgg16, model)

    for animal in animals:
        os.makedirs(os.path.join(output_dir_path, animal))

    for image_path, prediction in predictions.items():
        shutil.copy(image_path, os.path.join(output_dir_path, prediction, os.path.basename(image_path)))
    
    
    class_prediction_counts = {
        animal: len(os.listdir(os.path.join(output_dir_path, animal))) for animal in animals
    }
    
    log.info(f"Ejecución exitosa")
    
    print(12*"-"+"Classifier results"+12*"-")
    total = sum(class_prediction_counts.values())
    for animal, count in class_prediction_counts.items():
        print(f"{animal}: {count} ({100*count/total:.2f}%)")


    

if __name__ == "__main__":
    main()