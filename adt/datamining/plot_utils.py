import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import mpld3

def generate_plot_html():
    # Establish connection
    conn = psycopg2.connect(dbname="postgres", user="postgres", password="postgres", host="db", port="5432")

    # SQL query
    query = """
        SELECT id, model_year, make, car_model, vehicle_class, engine_size, cylinders, transmission,
               fuel_type, city, highway, combined, combined_mpg, "CO2_Emission",
               motor, "city_kWh", "highway_kWh", "combined_kWh", range, recharge_time, 
               fuel_type2, range2, "combined_PHEV", vehicle_type
        FROM datamining_carinfomodel;
    """

    df = pd.read_sql_query(query, conn)

    # Calculate mean CO2 emissions brand-wise
    mean_co2_by_brand = df.groupby('make')['CO2_Emission'].mean()

    # Sort the mean CO2 emissions in descending order
    sorted_mean_co2 = mean_co2_by_brand.sort_values(ascending=False)

    top_10 = sorted_mean_co2.head(10)
    bottom_10 = sorted_mean_co2.tail(10)

    # Plotting the horizontal bar graph for top 10
    plt.figure(figsize=(6, 5))
    top_10.plot(kind='barh', color='skyblue')
    plt.title('Most 10 Mean CO2 Emissions by Brand')
    plt.xlabel('Mean CO2 Emissions (g/km)')
    plt.ylabel('Brand')
    plt.tight_layout()
    top_html = mpld3.fig_to_html(plt.gcf())
    plt.close()

    # Plotting the horizontal bar graph for bottom 10
    plt.figure(figsize=(6, 5))
    bottom_10.plot(kind='barh', color='salmon')
    plt.title('LEast 10 Mean CO2 Emissions by Brand')
    plt.xlabel('Mean CO2 Emissions (g/km)')
    plt.ylabel('Brand')
    plt.tight_layout()
    bottom_html = mpld3.fig_to_html(plt.gcf())
    plt.close()

    # Group by 'Fuel type' and 'Model year', then calculate the average CO2 emissions
    conventional_df = df[df['vehicle_type'] == "Conventional"]
    co2_emission_yearly = conventional_df.groupby(['fuel_type', 'model_year'])['CO2_Emission'].mean().reset_index()
    fuel_type_labels = {'X': 'Regular gasoline', 'Z': 'Premium gasoline', 'D': 'Diesel', 'E': 'E85', 'N': 'Natural Gas'}
    co2_emission_yearly['fuel_type'] = co2_emission_yearly['fuel_type'].map(fuel_type_labels)
    co2_emission_yearly = co2_emission_yearly.groupby(['fuel_type', 'model_year']).mean().reset_index()
    co2_emission_yearly_pivot = co2_emission_yearly.pivot(index='model_year', columns='fuel_type', values='CO2_Emission')

    # Plot the data
    plt.figure(figsize=(10, 6))
    co2_emission_yearly_pivot.plot(kind='line', marker='o')
    plt.title('Average CO2 Emissions by Fuel Type and Year')
    plt.xlabel('Model Year')
    plt.ylabel('Average CO2 Emissions (g/km)')
    plt.legend(title='Fuel Type')
    plt.tight_layout()
    co2_emission_html = mpld3.fig_to_html(plt.gcf())
    plt.close()

    # Combine HTMLs
    plot_html = {
        "top_10": top_html,
        "bottom_10": bottom_html,
        "co2_emission_yearly": co2_emission_html
    }

    return plot_html
