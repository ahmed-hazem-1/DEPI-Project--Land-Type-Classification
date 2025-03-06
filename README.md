# Land Type Classification using Sentinel-2 Satellite Images

## Project Overview

This project focuses on classifying different land types (agriculture, water, urban areas, desert, roads, trees) using Sentinel-2 satellite images. By leveraging Deep Neural Networks (DNNs), we aim to develop an accurate and efficient model for land classification, which can be beneficial for environmental monitoring, urban planning, and agricultural management.

## Purpose & Real-World Benefits

The primary purpose of this project is to provide an automated and reliable solution for land classification using satellite imagery. This technology can be leveraged by:

- **Urban Planners**: To monitor land usage and optimize city expansion.
- **Environmental Agencies**: To track deforestation, water bodies, and climate change effects.
- **Agricultural Experts**: To analyze soil health and vegetation coverage for better crop management.
- **Disaster Response Teams**: To assess land changes after natural disasters such as floods and wildfires.
- **Government & Policy Makers**: To make data-driven decisions for sustainable land development and resource allocation.

By implementing an AI-driven classification model, we reduce the time and manual effort required for land classification while increasing accuracy and scalability.

## Dataset

We utilize Sentinel-2 satellite imagery, a high-resolution dataset provided by the European Space Agency (ESA). These images contain multiple spectral bands that capture detailed information about land surfaces, allowing for precise classification.

### Data Preprocessing

1. **Image Acquisition**: Downloading and selecting relevant Sentinel-2 images.
2. **Preprocessing**:
   - Resizing images to a standard resolution.
   - Normalizing pixel values for better model performance.
   - Extracting relevant spectral bands.
3. **Labeling**: Assigning land type labels to images based on geographic information and reference datasets.

## Methodology

1. **Exploratory Data Analysis (EDA)**:
   - Visualizing different land types.
   - Analyzing spectral characteristics of land classes.
2. **Feature Engineering**:
   - Extracting essential spectral bands for classification.
   - Applying Principal Component Analysis (PCA) if necessary.
3. **Model Training**:
   - Using a Deep Neural Network (DNN) to classify images.
   - Experimenting with CNN architectures for improved accuracy.
   - Training with labeled datasets and evaluating performance.
4. **Evaluation Metrics**:
   - Accuracy
   - Precision, Recall, and F1-score
   - Confusion Matrix Analysis

## Tools & Technologies

- **Python**: Primary programming language.
- **TensorFlow/Keras**: Deep learning framework for building the classification model.
- **OpenCV**: Image processing and manipulation.
- **GDAL**: Handling geospatial data.
- **Pandas, NumPy**: Data manipulation and analysis.
- **Matplotlib, Seaborn**: Visualization of data and results.

## Challenges & Solutions

- **Handling Large Datasets**: Used cloud storage and batch processing to manage high-resolution images efficiently.
- **Class Imbalance**: Applied data augmentation techniques to balance training samples.
- **Feature Selection**: Experimented with different spectral bands to find the most informative ones.

## Future Improvements

- Integrate more advanced deep learning techniques like Transformer models for enhanced accuracy.
- Expand dataset to include multi-seasonal images for better generalization.
- Deploy the model as a web application for easy accessibility.

## Contributors

- **Ahmed Hazem Elabady** - Team Leader, Model Selection, Model Development, Final Report & Presentation.
- **Ahmed Selim** - Data Collection, Image Quality Validation, Model Training & Optimization.
- **Mustafa Bayomi** - Exploratory Data Analysis (EDA), Preprocessing, Feature Analysis.
- **John George** - API Development using Flask/FastAPI, Dashboard Integration Support.
- **Mohamed Yasser** - Deployment Monitoring, Power BI Dashboard Development.



