from tensorflow.keras.preprocessing import image
from tensorflow import keras
import cv2
import numpy as np

classes = {
    0:"mancha angular",
    1:"Ferrugem (Uromyces phaseoli)",
    2: "Nenhuma",
}

def get_class_label(predicted_class):
    if predicted_class == 0:
        return classes[0]
    if predicted_class == 1:
        return classes[1]
    else:
        return classes[2]
    
def classify_image(image_path):
    image = cv2.imread("media/"+image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    image = image / 255.0

    image = np.array([image])

    model = keras.models.load_model('/Users/marcelocolares/Documents/Bean_Leaf_Lesions_Classification /bean_leaf_classification/models/bean_leaf_lesion.keras') 

    prediction = model.predict(image)

    prediction = np.argmax(prediction,axis=1)

    prediction = get_class_label(prediction)

    return prediction

def get_prompt(label):
    
    if label==classes[0]:
        return """
        a definir 
        """
    if label == classes[1]:
        return """
        minha folha de feijão possui ferrugem  quais as causas?
        """
    else:
        return """
        a definir
        """
    