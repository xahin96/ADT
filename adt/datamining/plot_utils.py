from .models import CarInfoModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import mpld3

# recommendation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
# spider
import plotly.graph_objects as go

import io
import base64

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

    # # Plotting the horizontal bar graph for top 10
    # plt.figure(figsize=(6, 5))
    # top_10.plot(kind='barh', color='skyblue')
    # plt.title('Top 10 Brands with the Highest Mean CO2 Emissions')
    # plt.xlabel('Mean CO2 Emissions (g/km)')
    # plt.ylabel('Brand')
    # plt.yticks(range(len(top_10)), top_10.index)
    # plt.tight_layout()
    # top_html = mpld3.fig_to_html(plt.gcf())
    # plt.close()


    # Plotting the horizontal bar graph for top 10
    plt.figure(figsize=(10, 8))
    top_10.plot(kind='barh', color='skyblue')
    plt.title('Top 10 Brands with the Highest Mean CO2 Emissions')
    plt.xlabel('Mean CO2 Emissions (g/km)')
    plt.ylabel('Brand')
    plt.yticks(range(len(top_10)), top_10.index)  # Set y-ticks to brand names
    plt.tight_layout()

    # Save the plot to a BytesIO object
    top_buffer = io.BytesIO()
    plt.savefig(top_buffer, format='png')
    plt.close()

    # Convert the BytesIO object to HTML
    top_buffer.seek(0)
    top_html = """
        <img src="data:image/png;base64,{}" style="width: 37vw;"/>
    """.format(base64.b64encode(top_buffer.read()).decode())

    # # Plotting the horizontal bar graph for bottom 10
    # plt.figure(figsize=(6, 5))
    # bottom_10.plot(kind='barh', color='salmon')
    # plt.title('Top 10 Brands with the Lowest Mean CO2 Emissions')
    # plt.xlabel('Mean CO2 Emissions (g/km)')
    # plt.ylabel('Brand')
    # plt.yticks(range(len(bottom_10)), bottom_10.index)
    # plt.tight_layout()
    # bottom_html = mpld3.fig_to_html(plt.gcf())
    # plt.close()

    # Plotting the horizontal bar graph for bottom 10
    plt.figure(figsize=(10, 8))
    bottom_10.plot(kind='barh', color='salmon')
    plt.title('Top 10 Brands with the Lowest Mean CO2 Emissions')
    plt.xlabel('Mean CO2 Emissions (g/km)')
    plt.ylabel('Brand')
    plt.yticks(range(len(bottom_10)), bottom_10.index)  # Set y-ticks to brand names
    plt.tight_layout()

    # Save the plot to a BytesIO object
    bottom_buffer = io.BytesIO()
    plt.savefig(bottom_buffer, format='png')
    plt.close()

    # Convert the BytesIO object to HTML
    bottom_buffer.seek(0)
    bottom_html = """
        <img src="data:image/png;base64,{}" style="width: 37vw;"/>
    """.format(base64.b64encode(bottom_buffer.read()).decode())

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



NUMBER_OF_ONLY_EV = 10
NUMBER_OF_HYBRID_WITH_EV = 6
NUMBER_OF_EV_WITH_HYBRID = 4
NUMBER_OF_CONVENTIONAL_WITH_HYBRID = 6
NUMBER_OF_HYBRID_WITH_CONVENTIONAL = 4
NUMBER_OF_RESULT = 20

def recommend_similar_cars(df, transformer, source_car_id, filter_carbon_emission=False):
    # print("-1"*30)
    # Get the source car from the DataFrame using its ID
    source_car = df[df['id'] == source_car_id].iloc[0]

    source_vehicle_type = source_car['vehicle_type']
    recommended_cars = {}

    source_car_index = df.index[df['id'] == source_car_id][0]

    if source_vehicle_type == "BEV":
        weighted_data = transformer.copy()
        user_weights = {'motor': 0.75, 'city_kWh': 0.7, 'highway_kWh': 0.8, 'combined_kWh': 0.9, 'range': 0.9, 'recharge_time': 0.9, 'CO2_Emission': 0.1}
        weighted_data[['motor', 'city_kWh', 'highway_kWh', 'combined_kWh','range', 'recharge_time', 'CO2_Emission']]*= user_weights
        similarity_scores = cosine_similarity(weighted_data.values, transformer.values)

        # Get indices of top similar cars
        # Exclude self-similarity and get top <NUMBER_OF_RESULTS> similar cars
        top_similar_indices = similarity_scores[source_car_index].argsort()[::-1][1:NUMBER_OF_ONLY_EV+1]

        # Filter and display recommended cars
        for idx in top_similar_indices:
            similar_car = df.iloc[idx]
            # print(f"0 BEV ID_df: {idx} -  ID_column: {similar_car[0]} {similar_car[1]} {similar_car[2]} ({similar_car[3]}), CO2 emissions: {similar_car['CO2_Emission']} g/km")

            # Skip if similar car has the same model as the source car
            if similar_car['car_model'] == source_car['car_model']:
                continue
            if similar_car['car_model'] not in recommended_cars or similar_car['model_year'] > recommended_cars[similar_car['car_model']][3]:
                recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])

    elif source_vehicle_type == "PHEV":

        weighted_data_hybrid = transformer.copy()
        user_weights_hybrid = {'engine_size': 0.8,'cylinders': 0.8,'city':0.8,'highway':0.8,'combined':0.85,'motor': 0.6,'range': 0.9, 'recharge_time': 0.8,'range2':0.85, 'CO2_Emission': 0.1}
        weighted_data_hybrid[['engine_size','cylinders','city','highway','combined','motor','range', 'recharge_time','range2', 'CO2_Emission']] *= user_weights_hybrid
        similarity_scores_hybrid = cosine_similarity(weighted_data_hybrid.values, transformer.values)

        top_similar_indices_hybrid = similarity_scores_hybrid[source_car_index].argsort()[::-1][1:NUMBER_OF_RESULT+1]

        weighted_data_electric =  transformer.copy()
        user_weights_electric= {'motor': 0.75, 'range': 0.9, 'recharge_time': 0.9, 'CO2_Emission': 0.1}
        weighted_data_electric[['motor','range', 'recharge_time','CO2_Emission']] *= user_weights_electric
        similarity_scores_electric = cosine_similarity(weighted_data_electric.values, transformer.values)

        # Filter indices to include only rows where CO2_Emission is 0
        filtered_indices_electric = [idx for idx in range(len(similarity_scores_electric[source_car_index])) if transformer.iloc[idx]['CO2_Emission'] == 0]

        # Sort and slice filtered indices to get top similar indices
        top_similar_indices_electric = sorted(filtered_indices_electric, key=lambda x: similarity_scores_electric[source_car_index][x], reverse=True)[1:NUMBER_OF_RESULT+1]

        counter = 0
        for idx in top_similar_indices_electric:
            if counter <= NUMBER_OF_EV_WITH_HYBRID:
                similar_car = df.iloc[idx]
                # print(f"1 BEV ID_df: {idx} -  ID_column: {similar_car[0]} {similar_car[1]} {similar_car[2]} ({similar_car[3]}), CO2 emissions: {similar_car['CO2_Emission']} g/km")
                recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])
                counter+=1


        counter = 0
        for idx in top_similar_indices_hybrid:
            if counter <= NUMBER_OF_HYBRID_WITH_EV:
                similar_car = df.iloc[idx]
                # print(f"2 PHEV ID_df: {idx} -  ID_column: {similar_car[0]} {similar_car[1]} {similar_car[2]} ({similar_car[3]}), CO2 emissions: {similar_car['CO2_Emission']} g/km")

                # Skip if similar car has the same model as the source car
                if similar_car['car_model'] == source_car['car_model']:
                    continue

                # Filter similar cars based on carbon emissions if filter_carbon_emission is True
                if filter_carbon_emission and similar_car['CO2_Emission'] > source_car['CO2_Emission']:
                    continue  # Skip if similar car has higher emissions

                if filter_carbon_emission:
                # Keep only the car with the least carbon emissions for each model
                    if similar_car['car_model'] not in recommended_cars or similar_car['CO2_Emission'] < recommended_cars[similar_car['car_model']][3]:
                        recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])
                        counter+=1
                else:
                    if similar_car['car_model'] not in recommended_cars or similar_car['model_year'] > recommended_cars[similar_car['car_model']][3]:
                        recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])
                        counter+=1
    
    elif source_vehicle_type == "Conventional":

        max_CO2_emission_limit = source_car['CO2_Emission']

        print(f"max CO2 : {max_CO2_emission_limit}")


        # Convesntional 
        weighted_data_conventional = transformer.copy()
        user_weights_conventional = {'engine_size': 0.6, 'cylinders': 0.55, 'city': 0.85, 'highway': 0.8, 'combined': 0.6, 'combined_mpg': 0.85, 'CO2_Emission': 0.1}
        weighted_data_conventional[['engine_size', 'cylinders', 'city', 'highway', 'combined', 'combined_mpg', 'CO2_Emission']] *= user_weights_conventional
        similarity_scores_conventional = cosine_similarity(weighted_data_conventional.values, transformer.values)

        top_similar_indices_conventional = similarity_scores_conventional[source_car_index].argsort()[::-1][1:NUMBER_OF_RESULT+1]


        # Hybrid
        weighted_data_hybrid = transformer.copy()
        user_weights_hybrid = {'engine_size': 0.8,'cylinders': 0.8,'city':0.8,'highway':0.8,'combined':0.85,'CO2_Emission': 1}
        weighted_data_hybrid[['engine_size','cylinders','city','highway','combined', 'CO2_Emission']] *= user_weights_hybrid

        similarity_scores_hybrid = cosine_similarity(weighted_data_hybrid.values, transformer.values)

        top_similar_indices_hybrid = similarity_scores_hybrid[source_car_index].argsort()[::-1][7000:10000+1]


        # add the conventional ones
        counter = 0
        for idx in top_similar_indices_conventional:
            if counter <= NUMBER_OF_CONVENTIONAL_WITH_HYBRID:
                similar_car = df.iloc[idx]

                if similar_car['CO2_Emission'] <= max_CO2_emission_limit:
                    # print(f"2 CONV ID_df: {idx} -  ID_column: {similar_car[0]} {similar_car[1]} {similar_car[2]} ({similar_car[3]}) {similar_car[4]}, CO2 emissions: {similar_car['CO2_Emission']} g/km")
                    # Skip if similar car has the same model as the source car
                    if similar_car['car_model'] == source_car['car_model']:
                        continue

                    # Filter similar cars based on carbon emissions if filter_carbon_emission is True
                    if filter_carbon_emission and similar_car['CO2_Emission'] > source_car['CO2_Emission']:
                        continue  # Skip if similar car has higher emissions

                    if filter_carbon_emission:
                    # Keep only the car with the least carbon emissions for each model
                        if similar_car['car_model'] not in recommended_cars or similar_car['CO2_Emission'] < recommended_cars[similar_car['car_model']][3]:
                            recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])
                            counter+=1
                    else:
                        if similar_car['car_model'] not in recommended_cars or similar_car['model_year'] > recommended_cars[similar_car['car_model']][3]:
                            recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])
                            counter+=1


        # add the hybrid ones
        counter = 0
        for idx in top_similar_indices_hybrid:
            if counter <= NUMBER_OF_HYBRID_WITH_CONVENTIONAL:
                similar_car = df.iloc[idx]

                if similar_car['CO2_Emission'] <= max_CO2_emission_limit:
                    # print(f"1 PHEV ID_df: {idx} -  ID_column: {similar_car[0]} {similar_car[1]} {similar_car[2]} ({similar_car[3]}) {similar_car[4]}, CO2 emissions: {similar_car['CO2_Emission']} g/km")
                    # Skip if similar car has the same model as the source car
                    if similar_car['car_model'] == source_car['car_model']:
                        continue

                    # Filter similar cars based on carbon emissions if filter_carbon_emission is True
                    if filter_carbon_emission and similar_car['CO2_Emission'] > source_car['CO2_Emission']:
                        continue  # Skip if similar car has higher emissions

                    if filter_carbon_emission:
                    # Keep only the car with the least carbon emissions for each model
                        if similar_car['car_model'] not in recommended_cars or similar_car['CO2_Emission'] < recommended_cars[similar_car['car_model']][3]:
                            recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])
                            counter+=1
                    else:
                        if similar_car['car_model'] not in recommended_cars or similar_car['model_year'] > recommended_cars[similar_car['car_model']][3]:
                            recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])
                            counter+=1

        
    # print(recommended_cars)
    return recommended_cars.values()


# SPIDER chart try older one direct from scalled data --> method from application
def plot_radar_chart(car1_id, car2_id, df, transformer):
    car1_whole_data = df[df['id'] == car1_id]
    car2_whole_data = df[df['id'] == car2_id]

    print(f" car 1 : {car1_whole_data}")
    print(f" car 2 : {car2_whole_data}")

    vehicle_type_car1 = car1_whole_data['vehicle_type'].iloc[0]
    vehicle_type_car2 = car2_whole_data['vehicle_type'].iloc[0]

    # When car1 -> convetional; car2 -> (should be) convetional or hybrid
    # When car1 -> hybrid; car2 -> (should be) hybrid or EV
    # When car1 -> EV; car2 -> (should be) EV

    print(vehicle_type_car1)
    print(vehicle_type_car2)
    
    attributes = [];
    if (vehicle_type_car1=="Conventional"):
        attributes = ['engine_size' ,'cylinders' , 'city' , 'highway' ,'combined' , 'combined_mpg' , 'CO2_Emission']
    elif (vehicle_type_car1=="BEV"):
        attributes = ['motor', 'range','city_kWh', 'highway_kWh', 'CO2_Emission', 'combined_kWh', 'recharge_time']
    elif (vehicle_type_car1=="PHEV"):
        if (vehicle_type_car2=="PHEV"):
            attributes = ['engine_size','cylinders','city','highway','combined','motor','range', 'recharge_time','range2', 'CO2_Emission']
        elif (vehicle_type_car2=="BEV"):
            attributes = ['motor','range', 'recharge_time','CO2_Emission']

    print(attributes)

    # Get data for car1 and car2
    car1_data = transformer[transformer.id == car1_id][attributes].iloc[0]
    car2_data = transformer[transformer.id == car2_id][attributes].iloc[0]

    car1_title = f'{car1_whole_data["make"].iloc[0]}'
    car2_title = f'{car2_whole_data["make"].iloc[0]}'

    # Plot
    trace_car1 = go.Scatterpolar(
        r=car1_data.values,
        theta=attributes,
        fill='toself',
        name=f'{car1_title}',
        # line=dict(color='red'),  # Adjust color as needed
    )

    trace_car2 = go.Scatterpolar(
        r=car2_data.values,
        theta=attributes,
        fill='toself',
        name=f'{car2_title}',
        # line=dict(color='blue'),  # Adjust color as needed
    )

    # Combine traces
    data = [trace_car1, trace_car2]

    # Define layout for the radar chart
    layout = go.Layout(
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(direction="clockwise"),
        ),
        showlegend=True,
        title=f'',
    )

    # Create figure
    fig = go.Figure(data=data, layout=layout)

    # Convert the plot to HTML
    plot_html = fig.to_html(full_html=False)

    return plot_html
