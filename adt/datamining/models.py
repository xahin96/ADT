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
    engine_size = models.FloatField(null=True, blank=True)
    cylinders = models.IntegerField(null=True, blank=True) # Cylinders
    transmission = models.CharField(null=True, blank=True, max_length=5) # Transmission
    fuel_type = models.CharField(null=True, blank=True, max_length=5) # Fuel Type
    city = models.FloatField(null=True, blank=True) #City (L/100 km)
    highway = models.FloatField(null=True, blank=True) #Highway (L/100 km)
    combined = models.FloatField(null=True, blank=True) #Combined (L/100 km)
    combined_mpg = models.IntegerField(null=True, blank=True) #Combined (mpg)
    CO2_Emission = models.IntegerField(null=True, blank=True) #CO2 emissions (g/km)
    CO2_Rating = models.IntegerField(null=True, blank=True) #CO2 rating
    smog_rating = models.IntegerField(null=True, blank=True) #Smog rating
    #EV HEV
    motor = models.IntegerField(null=True, blank=True) #Motor (kW)
    city_kWh = models.FloatField(null=True, blank=True) #City (kWh/100 km)
    highway_kWh = models.FloatField(null=True, blank=True) #Highway (kWh/100 km)
    combined_kWh = models.FloatField(null=True, blank=True) #Combined (kWh/100 km)
    range = models.IntegerField(null=True, blank=True) #Range 1 (km)
    recharge_time = models.IntegerField(null=True, blank=True) #Recharge time (h)
    fuel_type2 = models.CharField(null=True, blank=True, max_length=5) #Fuel type 2
    range2 = models.IntegerField(null=True, blank=True) #Range 2 (km)
    combined_PHEV = models.FloatField(null=True, blank=True) #Combined Le/100 km
    vehicle_type = models.CharField(max_length=15, choices=Vehicle_choices)
