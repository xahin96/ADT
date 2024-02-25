# datamining/views.py
import os

from django.shortcuts import render
import mpld3
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
from datamining.models import CarInfoModel


def datamining_page(request):
    # Your logic here
    # File paths for the datasets
    filepaths = [
        "datamining/data/MY2022 Fuel Consumption Ratings.csv",
        "datamining/data/MY2021 Fuel Consumption Ratings.csv",
        "datamining/data/MY2020 Fuel Consumption Ratings(1).csv",
        "datamining/data/MY2015-2019 Fuel Consumption Ratings.csv",
        "datamining/data/MY2023 Fuel Consumption Ratings.csv"
    ]

    # Reading each file and storing the DataFrame in a list
    dfs = [pd.read_csv(filepath, encoding='ISO-8859-1') for filepath in filepaths]

    # Combine all datasets into a single DataFrame
    all_data = pd.concat(dfs)

    # Calculating the average CO2 emissions for each make and model across the years
    co2_emissions = all_data.groupby(['Model year', 'Make', 'Model'])['CO2 emissions (g/km)'].mean().reset_index()

    # Pivot the data to have years as columns and makes and models as rows
    co2_emissions_pivot = co2_emissions.pivot_table(index=['Make', 'Model'], columns='Model year', values='CO2 emissions (g/km)')

    # Selecting 10 random make and model combinations
    random_selection = co2_emissions_pivot.sample(n=2, random_state=50)
# Create the Matplotlib plot
    plt.figure(figsize=(8, 5))
    for index, row in random_selection.iterrows():
        plt.plot(row.index, row.values, marker='o', label=f"{index[0]} {index[1]}")

    plt.subplots_adjust(right=0.6)
    plt.xlabel('Model Year')
    plt.ylabel('Average CO2 Emissions (g/km)')
    plt.title('Trend of CO2 Emissions by Make and Model Over the Years')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Convert the plot to HTML
    fig_html = mpld3.fig_to_html(plt.gcf())
    plt.close()

    # Pass the HTML to the template context
    context = {'fig_html': fig_html}

    return render(request, 'datamining/datamining_page.html', context)


def load_data(request):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    bev_directory = os.path.join(current_directory, 'data', 'BEV')
    phev_directory = os.path.join(current_directory, 'data', 'BEV')
    conventional_directory = os.path.join(current_directory, 'data', 'BEV')
    print(bev_directory)

    read_excel_and_insert_to_db(bev_directory, "BEV")
    read_excel_and_insert_to_db(phev_directory, "PHEV")
    read_excel_and_insert_to_db(conventional_directory, "Conventional")


def read_excel_and_insert_to_db(folder_path, v_type):
    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # Read CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)

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
