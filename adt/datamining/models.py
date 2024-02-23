from django.db import models
from djongo import models as djongo_models

Vehicle_choices = (
    ('Conventional','Conventional'),
    ('PHEV', 'PHEV'),
    ('BEV','BEV'),
)


class CarInfoModel(djongo_models.Model):
    model_year = models.IntegerField()
    make = models.CharField(max_length=50)
    car_model = models.CharField(max_length=50) 
    vehicle_class = models.CharField(max_length=50) 
    engine_size = models.FloatField()
    cylinders = models.IntegerField()
    transmission = models.CharField(max_length=5)
    fuel_type = models.CharField(max_length=5)
    city = models.FloatField()
    highway = models.FloatField()
    combined = models.FloatField()
    combined_mpg = models.IntegerField()
    CO2_Emission = models.IntegerField()
    CO2_Rating = models.IntegerField()
    smog_rating = models.IntegerField()
    motor = models.IntegerField()
    city_kWh = models.FloatField()
    highway_kWh = models.FloatField() 
    combined_kWh = models.FloatField() 
    range = models.IntegerField()
    recharge_time = models.IntegerField()
    fuel_type2 = models.CharField(max_length=5)
    range2 = models.IntegerField()
    combined_PHEV = models.FloatField()
    vehicle_type = models.CharField(max_length=15, choices=Vehicle_choices)


