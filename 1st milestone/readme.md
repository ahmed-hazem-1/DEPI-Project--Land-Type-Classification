# **First Milestone – Data Preparation & Augmentation**  

#### **📌 Overview**  
The first milestone of our project focused on **preparing, augmenting, and analyzing the EuroSAT dataset** to ensure it's well-structured for machine learning. This involved several key steps:  

✅ **Loading & Preprocessing Sentinel-2 Satellite Images**  
✅ **Applying Data Augmentation to Balance the Dataset**  
✅ **Computing & Visualizing NDVI (Vegetation Analysis)**  
✅ **Handling Dataset Issues & Debugging Errors**  

This document highlights **the challenges faced** and **how we solved them** throughout the process.  

---

## **🛠 Steps Taken & Challenges Faced**  

### **🔹 1. Loading & Preprocessing Data**  
At first, we needed to load and preprocess the **Sentinel-2 satellite images** stored in `.tif` format.  
#### **🚧 Challenge:**  
- Some images had **different dimensions** and needed resizing.  
- We had to **select the right spectral bands** for land classification.  

#### **✅ Solution:**  
- We resized all images to **64×64 pixels** for consistency.  
- We allowed flexibility to choose **either 4 bands (B2, B3, B4, B8).  

---

### **🔹 2. Data Augmentation to Balance the Dataset**  
The dataset originally had **an uneven number of images per category**, which could cause bias in model training.  
#### **🚧 Challenge:**  
- Some categories had **fewer than 4,000 images**, while others had more.  
- The augmentation needed to **stop exactly at 4,000 images per category** to avoid imbalance.  

#### **✅ Solution:**  
- We applied **random transformations** (flipping, rotation, brightness adjustment) to increase diversity.  
- A loop was implemented to **track the number of images per category** and stop augmenting once 4,000 images were reached.  

---

### **🔹 3. Computing & Visualizing NDVI**  
NDVI (Normalized Difference Vegetation Index) is a crucial metric for vegetation analysis. We computed it using:  
\[
NDVI = \frac{(NIR - Red)}{(NIR + Red + 1e-5)}
\]  
where **NIR = Band 8 (B8)** and **Red = Band 4 (B4)**.  

#### **🚧 Challenge:**  
- **Wrong band indexing** due to using only **4 bands instead of 13** caused indexing errors (`IndexError: index 7 is out of bounds`).  
- **NDVI visualization errors** because `X` was mistakenly stored as a file path instead of a NumPy array.  

#### **✅ Solution:**  
- Updated the **NDVI computation function** to correctly reference **B8 (index 3) and B4 (index 2)** when using 4 bands.  
- Ensured `X.npy` was **properly loaded** before computing NDVI.  

---

## **📊 Final Outcome**
By the end of this milestone, we successfully:  
✔ **Prepared the dataset** (resized, normalized, and structured)  
✔ **Balanced categories with augmentation** (4,000 images per class)  
✔ **Computed & visualized NDVI for vegetation analysis**  
✔ **Resolved dataset inconsistencies and fixed multiple errors**  

Now, with the **clean & well-structured dataset**, we are ready for **the next milestone: Model Training!** 🚀  
You can see the data at here [View](https://drive.google.com/drive/folders/1oZmDZHZLy0ILYLbXWYeT3K11CrbnVV0o?usp=sharing)

---

## **📌 Next Steps**
🔹 **Train model** using this dataset  
🔹 **Fine-tune augmentation & preprocessing for better accuracy**  


## **🎯 Contributions**  

This milestone was successfully completed through the collaborative efforts of the team, where each member played a crucial role in ensuring the dataset was well-prepared, augmented, and visualized effectively.  

### **🔹 Ahmed Selim**  
- Downloaded EuroSAT dataset and uploaded it to Drive
- Designed and implemented the **core algorithms** for **data preprocessing and NDVI computation**.  
- Developed the **visualization techniques** to analyze NDVI maps and dataset distribution.  
- Ensured the **accuracy of spectral band selection** for meaningful vegetation analysis.  

### **🔹 Ahmed Hazem**  
- Led the **data augmentation process**, ensuring **each category reached 4,000 images** while maintaining dataset quality.  
- Optimized **dataset structuring and storage**, making it easier to manage and access.  
- Suggested and implemented the **use of Google Drive for persistent dataset storage** and **Google Colab for scalable execution**, improving efficiency compared to local processing.  

Both **Ahmed Selim and Ahmed Hazem** worked together to troubleshoot errors, optimize workflows, and ensure the dataset was fully prepared for the next stage: **model training and evaluation**. 🚀  
