import numpy as np
import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model



def main():
    model_path = 'gru.sav'
    scaler_path = 'scaler.sav'

    try:
    # initialising the trained models
      model = joblib.load(model_path)
      standard_scaler = joblib.load(scaler_path)
    except Exception as e:
      st.error(f"Error loading models: {e}")
      return


    st.title('Bitcoin Predictor')
    st.subheader('Detect Defect Present', divider=True)
    st.markdown('The Prediction model will predict the Closing Prices of Bitcoin based on histroical data provided:')

    # Input for historical data size
    input_size = st.number_input("Enter the size of your historical data:", min_value=1, step=1, format="%d")
    
    if input_size>0:
      input_data = ([st.number_input("Enter the closing price: ") for _ in range(input_size)])

    
      # Display user-input data in a table
      input_df = pd.DataFrame({'Day': range(1, input_size + 1), 'Closing Price': input_data})
      st.dataframe(input_df)

      #convert to numpy array
      user_input = (np.array([input_data])).reshape(-1,1)

      # Scale the input 
      scaled_input = standard_scaler.transform(user_input)

      # prediction using the trained model
      predicted_close_scaled = model.predict(scaled_input)

      # Inverse transform to get the prediction in the original scale
      predicted_close = standard_scaler.inverse_transform(predicted_close_scaled)

      predicted_values = predicted_close[0,0]

      # Display predicted and actual values
      result_df = pd.DataFrame({'Day': range(1, input_size + 1),
                                'Predicted Closing Price': predicted_values})
      st.subheader('Actual vs Predicted Closing Prices:')
      st.dataframe(result_df)

if __name__ == '__main__':
    main()

