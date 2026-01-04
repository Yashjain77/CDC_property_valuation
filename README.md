# Satellite Imagery–Based Property Valuation  
### Multimodal Residual Fusion of Tabular Data and Satellite Imagery

---

## Project Overview

This project implements a **multimodal regression pipeline** for predicting property prices by
combining **structured tabular data** with **satellite imagery**.

Traditional property valuation models rely only on numerical attributes such as area, number of
rooms, and location coordinates. However, these features fail to capture important **neighborhood-
level characteristics** like road connectivity, urban density, and surrounding infrastructure.

To address this limitation, this project integrates satellite imagery using a **residual fusion
architecture**, where visual information refines predictions made by a strong tabular baseline.

---

## Objectives

- Predict property prices using a **multimodal learning framework**
- Programmatically acquire **satellite imagery** using latitude and longitude
- Perform **exploratory and geospatial data analysis**
- Extract neighborhood-level visual features using **CNNs**
- Compare **tabular-only vs multimodal models**
- Ensure **model explainability** using Grad-CAM

---

## Model Architecture

### 🔹 Tabular Branch
- Input: Property attributes (area, rooms, amenities, latitude, longitude)
- Model: **XGBoost Regressor**
- Output: Baseline log-price prediction

### 🔹 Image Branch
- Input: Satellite images (Zoom-16)
- Model: **Convolutional Neural Network**
- Output: Residual price correction

### 🔹 Fusion Strategy
- Final Prediction = Baseline Prediction + CNN Residual


This design prevents satellite imagery from overpowering strong tabular signals while allowing it
to add meaningful neighborhood-level context.

---

## Project Structure

```text
CDC_property_valuation/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── images/
│
├── notebooks/
│   ├── preprocess_tabular.ipynb
│   ├── model_training.ipynb
│   └── predict.ipynb
│
├── visualisation/
│   ├── eda.ipynb
│   ├── geospatial.ipynb
│   ├── grad_cam.ipynb
│   └── results_visualisation.ipynb
│
├── outputs/
│   ├── *.pth / *.pkl
│   ├── *.csv
│   └── metrics.json
│
├── src/
│   ├── dataset.py
│   ├── models.py
│   └── image_features.py
│
├── data_fetcher.py
├── requirements.txt
├── README.md
└── .gitignore

```

## Setup Instructions

### 1️ Clone the Repository :-
git clone ...

cd ...

### 2️ Create a Virtual Environment :- 
python -m venv .venv

### 3️ Activate the Virtual Environment (Windows) :- 
.venv\Scripts\Activate.ps1

### 4️ Install Dependencies :- 
pip install -r requirements.txt

### 5️ Fetch Satellite Images :- 
python data_fetcher.py

### 6️ Preprocess Tabular Data :- 
preprocessing.ipynb

### 7️ Train the Model :- 
model_training.ipynb

### 8️ Generate Predictions :- 
predict.ipynb

## Evaluation Metrics

- RMSE (Root Mean Squared Error)
- R² Score

### Models evaluated:

- Tabular-only baseline
- Naive multimodal fusion
- Residual fusion (proposed)

---

## Explainability

The contribution of satellite imagery is analyzed using Grad-CAM, which highlights spatial
regions in satellite images that influence price predictions.

This provides insights into economically meaningful features such as:

- Road networks
- Building density
- Urban layout

---

## Tech Stack

- Data Processing: Pandas, NumPy, GeoPandas
- Machine Learning: Scikit-learn, XGBoost
- Deep Learning: PyTorch
- Image Processing: OpenCV, PIL
- Visualization: Matplotlib, Seaborn
