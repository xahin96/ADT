# datamining/urls.py
from django.urls import path
from .views import datamining_page

urlpatterns = [
    path('', datamining_page, name='datamining_page'),
    path('load/', datamining_page, name='datamining_page'),
]
