from django.shortcuts import render
from django.http import HttpResponse


def home_page(request):
    return render(request, 'home.html')
def minha_nova_view(request):
    return render(request, 'home.html')