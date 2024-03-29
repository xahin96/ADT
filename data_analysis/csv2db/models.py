from django.db import models
from djongo import models as djongo_models

Vehicle_choices = (
    ('Conventional','Conventional'),
    ('PHEV', 'PHEV'),
    ('BEV','BEV'),
)


class CarInfoModel(djongo_models.Model):
    model_year = models.IntegerField() # Model year
    make = models.CharField(max_length=50) # Make
    car_model = models.CharField(max_length=50) #Model
    vehicle_class = models.CharField(max_length=50) #Vehicle class
    engine_size = models.FloatField() # Engine size (L)
    cylinders = models.IntegerField() # Cylinders
    transmission = models.CharField(max_length=5) # Transmission
    fuel_type = models.CharField(max_length=5) # Fuel Type
    city = models.FloatField() #City (L/100 km)
    highway = models.FloatField() #Highway (L/100 km)
    combined = models.FloatField() #Combined (L/100 km)
    combined_mpg = models.IntegerField() #Combined (mpg)
    CO2_Emission = models.IntegerField() #CO2 emissions (g/km)
    CO2_Rating = models.IntegerField() #CO2 rating
    smog_rating = models.IntegerField() #Smog rating
    #EV HEV
    motor = models.IntegerField() #Motor (kW)
    city_kWh = models.FloatField() #City (kWh/100 km)
    highway_kWh = models.FloatField() #Highway (kWh/100 km)
    combined_kWh = models.FloatField() #Combined (kWh/100 km)
    range = models.IntegerField() #Range 1 (km)
    recharge_time = models.IntegerField() #Recharge time (h)
    fuel_type2 = models.CharField(max_length=5) #Fuel type 2
    range2 = models.IntegerField() #Range 2 (km)
    combined_PHEV = models.FloatField() #Combined Le/100 km
    vehicle_type = models.CharField(max_length=15, choices=Vehicle_choices)
