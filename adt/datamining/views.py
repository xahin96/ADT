# datamining/views.py
from django.shortcuts import render
import mpld3
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd



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