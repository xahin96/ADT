import pandas as pd
import os
import django

# Set up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "your_project_name.settings")
django.setup()

# Import your model
from your_app_name.models import CarInfoModel

def read_excel_and_insert_to_db(folder_path, v_type):
    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file_name)
            # Read Excel file into a pandas DataFrame
            df = pd.read_excel(file_path)

            # Iterate over each row in the DataFrame
            for index, row in df.iterrows():
                # Create a new instance of your Django model
                car_info = CarInfoModel(
                    model_year=row['Model year'],
                    make=row['Make'],
                    car_model=row['Model'],
                    vehicle_class=row['Vehicle class'],
                    engine_size=row['Engine size (L)'],
                    cylinders=row['Cylinders'],
                    transmission=row['Transmission'],
                    fuel_type=row['Fuel Type'],
                    city=row['City (L/100 km)'],
                    highway=row['Highway (L/100 km)'],
                    combined=row['Combined (L/100 km)'],
                    combined_mpg=row['Combined (mpg)'],
                    CO2_Emission=row['CO2 emissions (g/km)'],
                    CO2_Rating=row['CO2 rating'],
                    smog_rating=row['Smog rating'],
                    motor=row['Motor (kW)'],
                    city_kWh=row['City (kWh/100 km)'],
                    highway_kWh=row['Highway (kWh/100 km)'],
                    combined_kWh=row['Combined (kWh/100 km)'],
                    range=row['Range 1 (km)'],
                    recharge_time=row['Recharge time (h)'],
                    fuel_type2=row['Fuel type 2'],
                    range2=row['Range 2 (km)'],
                    combined_PHEV=row['Combined Le/100 km'],
                    vehicle_type=v_type
                )
                # Save the instance to the database
                car_info.save()

# Call the function with the path to your folder containing Excel files
read_excel_and_insert_to_db("../../data/BEV", "BEV")
read_excel_and_insert_to_db("../../data/PHEV", "PHEV")
read_excel_and_insert_to_db("../../data/Conventional", "Conventional")
