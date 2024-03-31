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
    plt.title('Least 10 Mean CO2 Emissions by Brand')
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

NUMBER_OF_ONLY_EV = 10
NUMBER_OF_HYBRID_WITH_EV = 6
NUMBER_OF_EV_WITH_HYBRID = 4
NUMBER_OF_CONVENTIONAL_WITH_HYBRID = 6
NUMBER_OF_HYBRID_WITH_CONVENTIONAL = 4
NUMBER_OF_RESULT = 100

def recommend_similar_cars(df, transformer, source_car_id, filter_carbon_emission=False):
    # print("-1"*30)
    # Get the source car from the DataFrame using its ID
    source_car = df[df['id'] == source_car_id].iloc[0]

    source_vehicle_type = source_car['vehicle_type']
    recommended_cars = {}

    source_car_index = df.index[df['id'] == source_car_id][0]
    # print("-2"*30)

    if source_vehicle_type == "BEV":
        # print("-3"*30)
        try:
            weighted_data = transformer.copy()
        except Exception as e:

            # print("-3a"*30)
            print(e)
        user_weights = {'motor': 0.75, 'city_kWh': 0.7, 'highway_kWh': 0.8, 'combined_kWh': 0.9, 'range': 0.9, 'recharge_time': 0.9, 'CO2_Emission': 0.1}
        weighted_data[['motor', 'city_kWh', 'highway_kWh', 'combined_kWh','range', 'recharge_time', 'CO2_Emission']]*= user_weights
        similarity_scores = cosine_similarity(weighted_data.values, transformer.values)
        # print("-3a"*30)
        # Get indices of top similar cars
        # Exclude self-similarity and get top <NUMBER_OF_RESULTS> similar cars
        top_similar_indices = similarity_scores[source_car_index].argsort()[::-1][1:NUMBER_OF_ONLY_EV+1]
        # print("-3c"*30)
        # Filter and display recommended cars
        for idx in top_similar_indices:
            similar_car = df.iloc[idx]
            print(f"0 BEV ID_df: {idx} -  ID_column: {similar_car[0]} {similar_car[1]} {similar_car[2]} ({similar_car[3]}), CO2 emissions: {similar_car['CO2_Emission']} g/km")

            # Skip if similar car has the same model as the source car
            if similar_car['car_model'] == source_car['car_model']:
                continue
            if similar_car['car_model'] not in recommended_cars or similar_car['model_year'] > recommended_cars[similar_car['car_model']][3]:
                recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])
        # print("-3b"*30)

    elif source_vehicle_type == "PHEV":
        # print("-4"*30)

        weighted_data_hybrid = transformer.copy()
        user_weights_hybrid = {'engine_size': 0.8,'cylinders': 0.8,'city':0.8,'highway':0.8,'combined':0.85,'motor': 0.6,'range': 0.9, 'recharge_time': 0.8,'range2':0.85, 'CO2_Emission': 0.1}
        weighted_data_hybrid[['engine_size','cylinders','city','highway','combined','motor','range', 'recharge_time','range2', 'CO2_Emission']] *= user_weights_hybrid
        similarity_scores_hybrid = cosine_similarity(weighted_data_hybrid.values, transformer.values)

        top_similar_indices_hybrid = similarity_scores_hybrid[source_car_index].argsort()[::-1][1:NUMBER_OF_HYBRID_WITH_EV+1]

        weighted_data_electric =  transformer.copy()
        user_weights_electric= {'motor': 0.75, 'range': 0.9, 'recharge_time': 0.9, 'CO2_Emission': 0.1}
        weighted_data_electric[['motor','range', 'recharge_time','CO2_Emission']] *= user_weights_electric
        similarity_scores_electric = cosine_similarity(weighted_data_electric.values, transformer.values)

        # Filter indices to include only rows where CO2_Emission is 0
        filtered_indices_electric = [idx for idx in range(len(similarity_scores_electric[source_car_index])) if transformer.iloc[idx]['CO2_Emission'] == 0]

        # Sort and slice filtered indices to get top similar indices
        top_similar_indices_electric = sorted(filtered_indices_electric, key=lambda x: similarity_scores_electric[source_car_index][x], reverse=True)[1:NUMBER_OF_EV_WITH_HYBRID+1]


        for idx in top_similar_indices_electric:
            similar_car = df.iloc[idx]
            print(f"1 BEV ID_df: {idx} -  ID_column: {similar_car[0]} {similar_car[1]} {similar_car[2]} ({similar_car[3]}), CO2 emissions: {similar_car['CO2_Emission']} g/km")
            recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])


        for idx in top_similar_indices_hybrid:
            similar_car = df.iloc[idx]
            print(f"2 PHEV ID_df: {idx} -  ID_column: {similar_car[0]} {similar_car[1]} {similar_car[2]} ({similar_car[3]}), CO2 emissions: {similar_car['CO2_Emission']} g/km")

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
            else:
                if similar_car['car_model'] not in recommended_cars or similar_car['model_year'] > recommended_cars[similar_car['car_model']][3]:
                    recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])
    
    elif source_vehicle_type == "Conventional":
        # print("-5"*30)

        max_CO2_emission_limit = source_car['CO2_Emission']

        print(f"max CO2 : {max_CO2_emission_limit}")

        # TODO add logic

        # Convesntional 
        weighted_data_conventional = transformer.copy()
        user_weights_conventional = {'engine_size': 0.6, 'cylinders': 0.55, 'city': 0.85, 'highway': 0.8, 'combined': 0.6, 'combined_mpg': 0.85, 'CO2_Emission': 0.1}
        weighted_data_conventional[['engine_size', 'cylinders', 'city', 'highway', 'combined', 'combined_mpg', 'CO2_Emission']] *= user_weights_conventional
        similarity_scores_conventional = cosine_similarity(weighted_data_conventional.values, transformer.values)

        top_similar_indices_conventional = similarity_scores_conventional[source_car_index].argsort()[::-1][1:NUMBER_OF_RESULT+1]

        # Hybrid
        weighted_data_hybrid = transformer.copy()
        user_weights_hybrid = {'engine_size': 0.8,'cylinders': 0.8,'city':0.8,'highway':0.8,'combined':0.85,'motor': 0.6,'range': 0.9, 'recharge_time': 0.8,'range2':0.85, 'CO2_Emission': 0.1}
        weighted_data_hybrid[['engine_size','cylinders','city','highway','combined','motor','range', 'recharge_time','range2', 'CO2_Emission']] *= user_weights_hybrid
        similarity_scores_hybrid = cosine_similarity(weighted_data_hybrid.values, transformer.values)

        top_similar_indices_hybrid = similarity_scores_hybrid[source_car_index].argsort()[::-1][1:NUMBER_OF_RESULT+1]

        # add the hybrid ones
        for idx in top_similar_indices_hybrid:
            similar_car = df.iloc[idx]

            if similar_car['CO2_Emission'] <= max_CO2_emission_limit:
                print(f"1 PHEV ID_df: {idx} -  ID_column: {similar_car[0]} {similar_car[1]} {similar_car[2]} ({similar_car[3]}) {similar_car[4]}, CO2 emissions: {similar_car['CO2_Emission']} g/km")
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
                else:
                    if similar_car['car_model'] not in recommended_cars or similar_car['model_year'] > recommended_cars[similar_car['car_model']][3]:
                        recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])

        # add the conventional ones
        for idx in top_similar_indices_conventional:
            similar_car = df.iloc[idx]

            if similar_car['CO2_Emission'] <= max_CO2_emission_limit:
                print(f"2 CONV ID_df: {idx} -  ID_column: {similar_car[0]} {similar_car[1]} {similar_car[2]} ({similar_car[3]}) {similar_car[4]}, CO2 emissions: {similar_car['CO2_Emission']} g/km")
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
                else:
                    if similar_car['car_model'] not in recommended_cars or similar_car['model_year'] > recommended_cars[similar_car['car_model']][3]:
                        recommended_cars[similar_car['car_model']] = (similar_car['id'], similar_car['make'], similar_car['car_model'], similar_car['model_year'], similar_car['CO2_Emission'])
    
    # print("-6"*30)

    # print(recommended_cars)
    return recommended_cars.values()

# SPIDER chart try older one direct from scalled data

# def plot_radar_chart(car1_id, car2_id, cars):

    #  Convert Django QuerySet to DataFrame
    # data = list(cars.values())
    # df = pd.DataFrame(data)
    #
    # # Accessing fields using dot notation
    # car1 = CarInfoModel.objects.get(id=car1_id)
    # car2 = CarInfoModel.objects.get(id=car2_id)
    #
    # vehicle_type_car1 = car1.vehicle_type
    # vehicle_type_car2 = car2.vehicle_type
    #
    # # List of attributes to be scaled
    # attributes_to_scale = ['engine_size', 'cylinders', 'city', 'highway', 'combined', 'combined_mpg', 'motor', 'city_kWh', 'highway_kWh', 'combined_kWh', 'range', 'range2', 'recharge_time', 'CO2_Emission']
    #
    # # Initialize MinMaxScaler
    # scaler = MinMaxScaler()
    #
    # # Scale the specified attributes
    # df_scaled = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    # df_scaled[attributes_to_scale] = scaler.fit_transform(df_scaled[attributes_to_scale])
    #
    # # Apply one-hot encoding
    # transformer = pd.get_dummies(df_scaled)
    #
    # transformer.fillna(0, inplace=True)
    #
    # # Constructing the title
    # make_car1 = getattr(car1, 'make', '')
    # car_model_car1 = getattr(car1, 'car_model', '')
    # make_car2 = getattr(car2, 'make', '')
    # car_model_car2 = getattr(car2, 'car_model', '')
    # title = f''
    #
    # # Print attributes
    # print(attributes_to_scale)
    #
    # # Determine attributes based on vehicle type
    # if vehicle_type_car1 == "Conventional":
    #     attributes = ['engine_size', 'cylinders', 'city', 'highway', 'combined', 'combined_mpg', 'CO2_Emission']
    # elif vehicle_type_car1 == "BEV":
    #     attributes = ['motor', 'range', 'city_kWh', 'highway_kWh', 'CO2_Emission', 'combined_kWh', 'recharge_time']
    # elif vehicle_type_car1 == "PHEV":
    #     if vehicle_type_car2 == "PHEV":
    #         attributes = ['engine_size', 'cylinders', 'city', 'highway', 'combined', 'motor', 'range', 'recharge_time', 'range2', 'CO2_Emission']
    #     elif vehicle_type_car2 == "BEV":
    #         attributes = ['motor', 'range', 'recharge_time', 'CO2_Emission']
    #
    # print(attributes)
    #
    # # Get data for car1 and car2
    # car1_data = transformer[transformer.id == car1.id][attributes].iloc[0]
    # car2_data = transformer[transformer.id == car2.id][attributes].iloc[0]
    #
    # # Radar chart attributes
    # labels = np.array(attributes)
    # num_vars = len(labels)
    #
    # # Compute angle for each axis
    # angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    #
    # # Complete the loop
    # car1_data = np.concatenate((car1_data,[car1_data[0]]))
    # car2_data = np.concatenate((car2_data,[car2_data[0]]))
    # angles += angles[:1]
    #
    # # Plot
    # fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    #
    # ax.fill(angles, car1_data, color='red', alpha=0.25, label=f'Car {car1.id}')
    # ax.fill(angles, car2_data, color='blue', alpha=0.25, label=f'Car {car2.id}')
    #
    # # Add legend
    # ax.legend(loc='upper right', fontsize='medium')
    #
    # # Add labels
    # ax.set_yticklabels([])
    # ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(attributes)
    #
    # # Setting range for the radar chart
    # ax.set_ylim(0, max(max(car1_data), max(car2_data)) * 1.1)
    #
    # # Adding title
    # plt.title(title, size=20, color='black', y=1.1)
    #
    # # Improve aesthetics
    # plt.grid(True, linestyle='--', linewidth=0.5)
    #
    # # plt.show()
    #
    # spider_html = mpld3.fig_to_html(plt.gcf())
    # plt.close()
    #
    # return spider_html

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
