import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load model from pickle file
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('onehot_encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

with open('standard_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit app starts here
st.title("Car Price Prediction Application")

# Load dataset for dropdown options (for getting unique options for dropdowns)
carsdf = pd.read_csv('carsdheko.csv')

# Define relevant categorical and numerical columns
categorical_cols = ['Fuel Type', 'Body Type', 'Transmission Type', 'OEM', 'Car Model', 'Insurance Validity', 'City']
numerical_cols = ['Kilometers Driven', 'Number of Previous Owners', 'Engine Displacement', 'Mileage', 'Torque', 'Car Age', 'Max_Power']

# Fuel Type: Assuming 'Fuel Type' was one-hot encoded, create a dropdown of fuel types
fuel_types = carsdf['Fuel Type'].unique()
ft = st.selectbox('Fuel Type', fuel_types)

# Body Type
bt = st.selectbox('Body Type', carsdf['Body Type'].unique())

# Kilometers Driven
km_driven = st.slider('Kilometers Driven', min_value=0, max_value=int(carsdf['Kilometers Driven'].max()))

# Transmission Type
Transmission = st.selectbox('Transmission Type', carsdf['Transmission Type'].unique())

# Number of Previous Owners
ownerNo = st.selectbox('Number of Previous Owners', carsdf['Number of Previous Owners'].unique())

# Max Power
max_power = int(carsdf['Max_Power'].max())
max_power_value = st.slider('Max Power (Bhp)', min_value=0, max_value=max_power)

# OEM
oem = st.selectbox('OEM', carsdf['OEM'].unique())

# Car Model
model_name = st.selectbox('Car Model', carsdf['Car Model'].unique())

# Car Age (You can calculate this based on the current year and manufacturing year or use the provided data)
max_car_age = int(carsdf['Car Age'].max())
car_age = st.slider('Car Age (Years)', min_value=0, max_value=max_car_age)

# Insurance Validity
Insurance_validity = st.selectbox('Insurance Validity', carsdf['Insurance Validity'].unique())

# City
city = st.selectbox('City', carsdf['City'].unique())

# Mileage
max_mileage = int(carsdf['Mileage'].max())
mileage = st.slider('Mileage (km/l)', min_value=0, max_value=max_mileage)


# Engine Displacement
max_engine_disp = int(carsdf['Engine Displacement'].max())
engine_disp = st.slider('Engine Displacement (cc)', min_value=0, max_value=max_engine_disp)


# Torque
max_torque = int(carsdf['Torque'].max())
torque = st.slider('Torque (Nm)', min_value=0, max_value=max_torque)







# Create a DataFrame for user input
user_input = pd.DataFrame({

    'Fuel Type': [ft],
    'Body Type': [bt],
    'Transmission Type': [Transmission],
    'Number of Previous Owners': [ownerNo],
    'OEM': [oem],
    'Car Model': [model_name],
    'Insurance Validity': [Insurance_validity],
    'Kilometers Driven': [km_driven],
    'City':[city],
    'Engine Displacement':[engine_disp],
    'Mileage':[mileage],
    'Torque':[torque], 
    'Car Age':[car_age],
    'Max_Power':[max_power]
    })

# Apply One-Hot Encoding and Scaling (Make sure price column is excluded)
encoded_user_input = encoder.transform(user_input[categorical_cols])
encoded_user_input_df = pd.DataFrame(encoded_user_input, columns=encoder.get_feature_names_out(categorical_cols))
scaled_num = scaler.transform(user_input[numerical_cols])
scaled_num_df = pd.DataFrame(scaled_num, columns=numerical_cols)

# Combine the processed features
processed_input = pd.concat([encoded_user_input_df, scaled_num_df], axis=1)

# Ensure that the processed input has the same columns as the training data
missing_cols = set(model.feature_names_in_) - set(processed_input.columns)
for col in missing_cols:
    processed_input[col] = 0  # Add missing columns with 0 value (to align with training data)

processed_input = processed_input[model.feature_names_in_]

# Prediction
if st.button('Predict Price'):
    prediction = model.predict(processed_input)
    price = round(prediction[0], 2)  # Round to two decimal places
    st.success(f"Predicted Car Price: ₹ {price} lakhs")

    




