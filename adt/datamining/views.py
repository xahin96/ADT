# datamining/views.py
import os
import mpld3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from django.shortcuts import get_object_or_404
from django.core.paginator import (
    Paginator,
    EmptyPage,
    PageNotAnInteger
)
import random
from django.db import connection
from django.http import HttpResponse
from pathlib import Path
from django.conf import settings
from .models import CarInfoModel
from .plot_utils import generate_plot_html
from django.shortcuts import render

# recommendation
from .plot_utils import recommend_similar_cars

import numpy as np
import pandas as pd

# for plotting the heatmap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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


def reset_sequence(model):
    model.objects.all().delete()
    connection.cursor().execute(f"ALTER SEQUENCE {model._meta.db_table}_id_seq RESTART WITH 1")


def load_data(request):
    reset_sequence(CarInfoModel)

    data_dir = Path(settings.BASE_DIR) / 'datamining' / 'data'
    bev_folder = data_dir / "BEV"
    phev_folder = data_dir / "PHEV"
    conventional_folder = data_dir / "Conventional"

    read_excel_and_insert_to_db(bev_folder, "BEV")
    read_excel_and_insert_to_db(phev_folder, "PHEV")
    read_excel_and_insert_to_db(conventional_folder, "Conventional")

    html = """
            <div>
                <h2>Data insertion done</h2>
                <a href="/datamining/">CHECK</a>
            </div>
            """

    response = HttpResponse(html, content_type="text/html", status=200)

    return response


def read_excel_and_insert_to_db(folder_path, v_type):

    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # Read CSV file into a pandas DataFrame
            df = pd.read_csv(file_path, encoding='ISO-8859-1')

            columns_to_check = [
                'Highway (L/100 km)',
                'Combined (L/100 km)',
                'Motor (kW)',
                'City (kWh/100 km)',
                'Highway (kWh/100 km)',
                'Combined (kWh/100 km)',
                'combined_PHEV',
                'smog_rating'  # Add 'smog_rating' to columns_to_check
            ]

            for column_name in columns_to_check:
                if column_name not in df.columns:
                    df[column_name] = None

            # Check if the 'Fuel type' column exists
            fuel_type_column = 'Fuel type 1' if v_type == 'PHEV' else 'Fuel type'
            if fuel_type_column not in df.columns:
                df[fuel_type_column] = None

            # Iterate over each row in the DataFrame
            for index, row in df.iterrows():
                # Replace NaN values with None
                row = row.replace({np.nan: None})

                # Split the combined_PHEV value by space
                combined_phev_value = row['combined_PHEV'].split()[0] if isinstance(row['combined_PHEV'], str) else None
                combined_kwh_value = row['combined_PHEV'].split('(')[-1].split(' ')[0] if isinstance(row['combined_PHEV'], str) else None

                # Create a new instance of your Django model
                car_info = CarInfoModel(
                    model_year=row['Model year'],
                    make=row['Make'],
                    car_model=row['Model'],
                    vehicle_class=row['Vehicle class'],
                    engine_size=row.get('Engine size (L)'),
                    cylinders=row.get('Cylinders'),
                    transmission=row.get('Transmission'),
                    fuel_type=row[fuel_type_column],
                    fuel_type2=row.get('Fuel type 2') if v_type == "PHEV" else None,
                    city=row.get('City (L/100 km)'),
                    highway=row.get('Highway (L/100 km)'),
                    combined=row.get('Combined (L/100 km)'),
                    combined_mpg=row.get('Combined (mpg)'),
                    CO2_Emission=row.get('CO2 emissions (g/km)'),
                    CO2_Rating=row.get('CO2 rating'),
                    smog_rating=row.get('Smog rating'),  # Ensure 'Smog rating' column exists in CSV
                    motor=row.get('Motor (kW)'),
                    city_kWh=row.get('City (kWh/100 km)'),
                    highway_kWh=row.get('Highway (kWh/100 km)'),
                    combined_kWh=row.get('Combined (kWh/100 km)'),
                    range=row.get('Range (km)') if v_type == "BEV" else row.get('Range 1 (km)') if v_type == "PHEV" else None,
                    recharge_time=row.get('Recharge time (h)'),
                    range2=row.get('Range 2 (km)'),
                    combined_PHEV=combined_phev_value,  # Assign split value to combined_PHEV
                    vehicle_type=v_type
                )
                # Save the instance to the database
                car_info.save()


# views.py
def dashboard(request):
    plot_html = generate_plot_html()

    return render(
        request,
        'datamining/dashboard.html',
        {
            'top_html': plot_html['top_10'],  # Corrected key
            'bottom_html': plot_html['bottom_10'],  # Corrected key
            'co2_emission_html': plot_html['co2_emission_yearly']  # Corrected key
        }
    )



def select_car(request):
    car_infos = CarInfoModel.objects.all()

    make = request.GET.get('make')
    model_year = request.GET.get('model_year')

    if make:
        car_infos = car_infos.filter(make__icontains=make)
    if model_year:
        car_infos = car_infos.filter(model_year=model_year)

    # Paginate the filtered queryset
    paginator = Paginator(car_infos, 10)  # 10 items per page
    page = request.GET.get('page')
    try:
        car_infos = paginator.page(page)
    except PageNotAnInteger:
        car_infos = paginator.page(1)
    except EmptyPage:
        car_infos = paginator.page(paginator.num_pages)
    return render(request, 'datamining/select-car.html', {'car_infos': car_infos})


def car_details(request, car_id):
    car_info = get_object_or_404(CarInfoModel, id=car_id)

    car_year = car_info.model_year

    # Retrieve all cars data from CarInfoModel
    all_cars = CarInfoModel.objects.filter(model_year__range=(car_year-3, car_year+3))   

    # Convert Django QuerySet to DataFrame
    data = list(all_cars.values())
    df = pd.DataFrame(data)

    # Preprocessing

    # Saving IDs
    ids = df['id'].tolist()

    # Each variable is converted in as many 0/1 variables as there are different values.
    # Columns in the output are each named after a value; if the input is a DataFrame, the name of the original variable is prepended to the value.
    transformer = pd.get_dummies(df)
    scaler = MinMaxScaler()
    scale_columns = ['engine_size', 'cylinders', 'city' , 'highway','combined', 'combined_mpg', 'CO2_Emission']
    transformer[scale_columns] = scaler.fit_transform(transformer[scale_columns])
    transformer.fillna(0, inplace=True)

    # Adding IDs back
    transformer['id'] = ids

    similar_cars = recommend_similar_cars(df, transformer, car_id)

    # print(similar_cars)
    id_list = [car[0] for car in similar_cars]
    # print(id_list)
    similar_cars = CarInfoModel.objects.filter(id__in=id_list)

    return render(request, 'datamining/car_details.html', {'car_info': car_info, 'similar_cars': similar_cars})


def compare_cars(request, car1_id, car2_id):
    car1 = get_object_or_404(CarInfoModel, id=car1_id)
    car2 = get_object_or_404(CarInfoModel, id=car2_id)
    return render(request, 'datamining/compare_cars.html', {'car1': car1, 'car2': car2})
