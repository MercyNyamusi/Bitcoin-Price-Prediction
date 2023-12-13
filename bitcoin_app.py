import numpy as np
import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

def main():
    st.title('Jijenge Bitcoin Price Predictor')
    image = Image.open('hero_image.png')
    st.image(image, caption='Bitcoin Predictor', width=600)
    st.subheader('Predict the Price of Bitcoin Based on Historical Data', divider=True)
    st.markdown('The Prediction model will predict the Closing Prices of Bitcoin based on historical data provided:')

    # Input for historical data size
    input_size = st.number_input("Enter the size of your historical data:", min_value=1, step=1, format="%d")

    # Initialize input_data with zeros
    input_data = [0] * input_size

    if input_size > 0:
        for i in range(input_size):
            data = st.number_input(f"Enter the closing price for day {i + 1}:", key=f"input_{i}")
            input_data[i] = data

        # Display user-input data in a table
        input_df = pd.DataFrame({'Day': range(1, input_size + 1), 'Closing Price': input_data})
        st.subheader('User Input Data:')
        st.dataframe(input_df)

        # Convert to numpy array
        user_input = np.array(input_data).reshape(-1, 1)

        # Scale the input
        scaler_path = 'scaler.sav'
        standard_scaler = joblib.load(scaler_path)
        scaled_input = standard_scaler.transform(user_input)

        # Load the model
        model_path = 'gru.sav'
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading the model: {e}")
            return

        # Prediction using the trained model
        predicted_close_scaled = model.predict(scaled_input)

        # Inverse transform to get the prediction in the original scale
        predicted_close = standard_scaler.inverse_transform(predicted_close_scaled)

        # Display predicted closing price
        st.subheader(f'Predicted Closing Price for day {input_size + 1}:')
        st.table(pd.DataFrame({'Predicted Closing Price': [predicted_close[0][0]]}))

if __name__ == '__main__':
    main()
