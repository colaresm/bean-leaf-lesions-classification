from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from utils.scripts import classify_image


def home_page(request):
    return render(request, 'home.html')

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('imagem'):
        image = request.FILES['imagem']

        fs = FileSystemStorage()

        filename = fs.save(image.name, image)

        uploaded_file_url = fs.url(filename)

        prediction = classify_image(image.name)

        print(image.name)
     
        return render(request, 'result.html', {'image_url':uploaded_file_url,'prediction': prediction})
    
    return render(request, 'result.html')