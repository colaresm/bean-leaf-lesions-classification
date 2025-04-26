from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras

def home_page(request):
    return render(request, 'home.html')

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('imagem'):
        imagem = request.FILES['imagem']
        fs = FileSystemStorage()
        filename = fs.save(imagem.name, imagem)
        uploaded_file_url = fs.url(filename)
       
        img = cv2.imread(imagem.name)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (224, 224))

        img = img / 255.0

        img = np.array([img])

        model = keras.models.load_model('/Users/marcelocolares/Documents/Bean_Leaf_Lesions_Classification /bean_leaf_classification/models/bean_leaf_lesion.keras') 

        pred = model.predict(img)

        pred = np.argmax(pred,axis=1)[0]
        print(pred)
        return render(request, 'result.html', {'uploaded_file_url': uploaded_file_url})
    
    return render(request, 'result.html')