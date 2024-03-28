from django.contrib import admin
from .models import CarInfoModel

class CarInfoModelAdmin(admin.ModelAdmin):
    list_display = ('model_year', 'make', 'car_model', 'vehicle_class', 'engine_size')
    search_fields = ['model_year', 'make', 'car_model', 'vehicle_class']
    list_filter = ['make', 'vehicle_class']

admin.site.register(CarInfoModel, CarInfoModelAdmin)
