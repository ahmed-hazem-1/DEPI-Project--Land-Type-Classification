# Land Type Classification - 2nd Milestone: Exploratory Data Analysis (EDA)

## 📌 Project Overview
This repository contains the second milestone of the **Land Type Classification using Sentinel-2 Satellite Images** project, which is part of the DEPI initiative. This milestone focuses on **Exploratory Data Analysis (EDA)**, aiming to extract insights, visualize distributions, and prepare data for machine learning modeling.

## 📂 Dataset
The dataset used in this milestone is the **EuroSAT Dataset**, which consists of Sentinel-2 satellite images categorized into 10 land types:

- **AnnualCrop**
- **Forest**
- **HerbaceousVegetation**
- **Highway**
- **Industrial**
- **Pasture**
- **PermanentCrop**
- **Residential**
- **River**
- **SeaLake**

The data is stored in `.npy` format and loaded from Google Drive.

## 🔧 Libraries Used
The following Python libraries are utilized in this milestone:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`
- `umap`
- `xgboost`
- `skimage`

## 📊 Exploratory Data Analysis (EDA) Steps
1. **Data Loading:** Importing features (`X.npy`), labels (`y.npy`), and vegetation index (`NDVI.npy`).
2. **Data Summary:** Checking dataset shapes and missing values.
3. **Feature Distribution:** Visualizing pixel intensity distributions and NDVI variations.
4. **Dimensionality Reduction:** Applying **PCA**, **t-SNE**, and **UMAP** for visualization.
5. **Texture Feature Extraction:** Utilizing `skimage` for Gray Level Co-occurrence Matrix (GLCM) analysis.
6. **Correlation Analysis:** Using heatmaps to understand feature relationships.
7. **Class Distribution:** Checking the balance of land type categories.

## 📈 Key Findings
- The dataset appears **well-balanced** across land types.
- NDVI values help distinguish vegetation-based classes.
- PCA and UMAP provide meaningful feature separations.
- Texture analysis offers additional feature engineering opportunities.

## 🚀 Next Steps
- Further feature engineering and transformation.
- Model training and hyperparameter tuning.
- Dashboard development for interactive visualization.

## 💡 How to Use
Clone the repository and open the notebook:
```bash
git clone https://github.com/ahmed-hazem-1/DEPI-Project--Land-Type-Classification.git
cd DEPI-Project--Land-Type-Classification
jupyter notebook 2nd_Milestone.ipynb
```
Ensure you have the required libraries installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn xgboost scikit-image
```

## 📬 Contributions
- Mustafa Bayomi – Performed the initial Exploratory Data Analysis (EDA) and data visualization.
- Ahmed Hazem – Reviewed and validated the EDA, refined the formatting, and improved the overall structure.
---
📌 *This milestone is part of the DEPI project under the supervision of MCIT Egypt.*

