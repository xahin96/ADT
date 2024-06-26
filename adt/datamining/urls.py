# datamining/urls.py
from django.urls import path
from .views import (
    datamining_page,
    load_data,
    dashboard,
    select_car,
    car_details,
    compare_cars
)

urlpatterns = [
    path('', dashboard, name='dashboard'),
    path('datamining_page', datamining_page, name='datamining_page'),
    path('load/', load_data, name='datamining_page'),
    path('select-car/', select_car, name='datamining_page'),
    path('car_details/<int:car_id>/', car_details, name='car_details'),
    path('compare_cars/<int:car1_id>/<int:car2_id>/', compare_cars, name='compare_cars'),
]
