# Land Type Classification Streamlit App

This interactive web application allows users to classify land types from satellite imagery using different models. Users can either upload images or select areas on a map for classification.

## Features

- **Multiple Model Support**: Choose between RGB, RGB+NIR, and NDVI models
- **Image Upload**: Upload satellite images in various formats (JPG, PNG, TIF, etc.)
- **Map Integration**: Select areas on an interactive map for classification
- **Analysis & Insights**: Get detailed statistics, visualizations, and insights about classified land types

## Setup Instructions

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

3. Access the app in your browser at http://localhost:8501

## Usage Guide

1. **Select a Model**:
   - Choose from RGB Model, RGB+NIR Model, or NDVI Model in the sidebar

2. **Input Method**:
   - **Upload Image**: Upload a satellite image file
   - **Map Selection**: Select coordinates on the map or input them manually

3. **Classification**:
   - Click "Classify Land Type" or "Fetch & Classify" to process the image
   - Review the results, including class probability, statistics, and insights

## Models

- **RGB Model**: Uses standard RGB channels for classification
- **RGB+NIR Model**: Incorporates Near-Infrared band for better vegetation analysis
- **NDVI Model**: Uses Normalized Difference Vegetation Index for specialized vegetation analysis

## Supported Land Types

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake
