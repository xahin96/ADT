# datamining/urls.py
from django.urls import path
from .views import datamining_page, load_data

urlpatterns = [
    path('', datamining_page, name='datamining_page'),
    path('load/', load_data, name='datamining_page'),
]
