# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from groq import Groq

# -----------------------------
# Function to load the model and label encoder
# -----------------------------
@st.cache_resource
def load_resources(model_path='poi_model.h5', encoder_path='label_encoder.joblib', feature_columns_path='feature_columns.npy'):
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        return None, None, None
    if not os.path.exists(encoder_path):
        st.error(f"Label Encoder file '{encoder_path}' not found.")
        return None, None, None
    if not os.path.exists(feature_columns_path):
        st.error(f"Feature Columns file '{feature_columns_path}' not found.")
        return None, None, None
    
    model = tf.keras.models.load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    feature_columns = np.load(feature_columns_path, allow_pickle=True).tolist()
    return model, label_encoder, feature_columns

# -----------------------------
# Function to preprocess user input
# -----------------------------
def preprocess_input(user_priorities, feature_columns):
    # Initialize a DataFrame with zeros
    user_feature_vector = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Set the appropriate priority columns to 1 based on user input
    for i, priority in enumerate(user_priorities, start=1):
        if priority != 'Unknown':
            column_name = f'PRIORITY_{i}_{priority}'
            if column_name in user_feature_vector.columns:
                user_feature_vector.at[0, column_name] = 1
    
    # If all priorities are 'Unknown', set 'PRIORITY_Unknown' to 1
    if all(priority == 'Unknown' for priority in user_priorities):
        if 'PRIORITY_Unknown' in user_feature_vector.columns:
            user_feature_vector.at[0, 'PRIORITY_Unknown'] = 1
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in user_feature_vector.columns:
            user_feature_vector[col] = 0
    
    # Reorder columns to match feature_columns
    user_feature_vector = user_feature_vector[feature_columns]
    
    return user_feature_vector.astype(float).values

# -----------------------------
# Function to get POI descriptions using Groq LLM
# -----------------------------
def get_poi_description(poi_name, groq_client):
    prompt = f"Provide a brief and engaging description for the Point of Interest: {poi_name}."
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant providing brief descriptions of Points of Interest."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )
        description = response.choices[0].message.content.strip()
        return description
    except Exception as e:
        return "Description not available."

# -----------------------------
# Streamlit App Layout
# -----------------------------
def main():
    st.set_page_config(page_title="POI Recommendation System", layout="wide")
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("📍 POI Recommendation System")
    st.markdown("### Discover Your Next Adventure Based on Your Priorities")
    
    # Load model, label encoder, and feature columns
    model, label_encoder, feature_columns = load_resources()
    
    if model is None or label_encoder is None or feature_columns is None:
        st.stop()
    
    # Define the priority options
    # Update this list based on your dataset's unique priority values
    priority_options = [
        'Adventure',
        'Scenic',
        'Relaxing',
        'Cultural',
        'Historical',
        'Family-friendly',
        'Nightlife',
        'Shopping',
        'Unknown'  # Ensure 'Unknown' is an option
    ]
    
    st.header("🎯 Define Your Priorities")
    
    # Collect user priorities for PRIORITY_1 to PRIORITY_5
    user_priorities = []
    cols = st.columns(5)
    for i in range(1, 6):
        with cols[i-1]:
            priority = st.selectbox(f"Priority {i}", options=priority_options, key=f'priority_{i}')
            user_priorities.append(priority)
    
    if st.button("🚀 Get Recommendations"):
        with st.spinner("Processing your input and generating recommendations..."):
            # Preprocess the input
            user_features = preprocess_input(user_priorities, feature_columns)
            
            # Make predictions
            predictions = model.predict(user_features)
            
            # Define the number of top recommendations
            N = 5  # You can make this configurable
            
            # Get the indices of the top N recommended POIs
            top_indices = np.argsort(predictions[0])[::-1][:N]
            
            # Decode the recommended POIs using the label encoder
            try:
                recommended_pois = label_encoder.inverse_transform(top_indices)
            except Exception as e:
                st.error(f"Error in decoding POIs: {e}")
                return
            
            # Get the corresponding probabilities
            recommended_probabilities = predictions[0][top_indices]
            
            # Initialize Groq client
            groq_api_key = "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"  # Replace with your actual Groq API key
            groq_client = Groq(
                api_key=groq_api_key,
            )
            
            # Fetch descriptions for each POI
            poi_details = []
            for poi in recommended_pois:
                description = get_poi_description(poi, groq_client)
                poi_details.append(description)
            
            # Display the recommended POIs with their probabilities and descriptions
            st.success("### Top Recommended POIs:")
            for idx, (poi, prob, desc) in enumerate(zip(recommended_pois, recommended_probabilities, poi_details), start=1):
                st.markdown(f"**{idx}. {poi}**")
                st.markdown(f"*Probability: {prob:.2f}*")
                st.markdown(f"{desc}")
                st.markdown("---")
    
    # Footer
    st.markdown(
        """
        <hr>
        <p style="text-align: center; color: gray;">
            © 2024 POI Recommendation System. All rights reserved.
        </p>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
