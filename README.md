# Crop Suitability and Yield Prediction Using Machine Learning

## Project Overview
This project is an end-to-end machine learning system designed to support crop suitability recommendation and yield prediction using soil, climate, and agronomic data. The primary motivation is to improve data-driven agricultural decision-making, with a particular focus on Ethiopian cereal crops.

The project follows a research-oriented yet production-ready structure, emphasizing reproducibility, explainability, and clear separation between experimentation and deployable code.

---

## Objectives
- Analyze soil and climate conditions relevant to major cereal crops  
- Handle agronomic data expressed as ranges (e.g., nutrient levels, rainfall, temperature)  
- Develop machine learning models for:
  - Crop suitability recommendation  
  - Crop species classification  
  - Yield prediction  
- Provide interpretable model outputs to support agronomic insights  
- Ensure full reproducibility using modern MLOps practices  

---

## Data Description
The dataset includes soil, climate, and agronomic variables such as:
- Soil nutrients (Nitrogen, Phosphorus, Potassium)
- Soil pH
- Temperature and rainfall
- Altitude and length of growing period (LGP)
- Crop type and crop species
- Yield ranges

Many variables are represented as ranges derived from agronomic studies. These ranges are explicitly handled during preprocessing and feature engineering rather than being treated as raw numeric values.

---

## Methodology
The project follows a structured machine learning workflow:
1. Exploratory Data Analysis (EDA) to understand distributions, relationships, and data quality  
2. Preprocessing and validation of range-based agronomic data  
3. Feature engineering for crop suitability and yield modeling  
4. Model training using classical and ensemble machine learning methods  
5. Model evaluation and explainability analysis  
6. Optional deployment through an API for practical use  

---

## Project Structure
The repository is organized to clearly separate concerns:

- `notebooks/` — exploratory analysis and experimentation  
- `src/` — production-ready code (ETL, preprocessing, modeling, evaluation)  
- `data/` — raw, processed, and feature-engineered datasets (DVC tracked)  
- `models/` — trained models and artifacts (DVC tracked)  
- `tests/` — unit tests for core components  
- `reports/` — figures, tables, and explainability outputs  

This structure supports both academic rigor and software engineering best practices.

---

## Reproducibility
Reproducibility is a core design principle of this project:
- Git is used for source code version control  
- DVC is used for dataset and model versioning  
- A Python virtual environment ensures dependency isolation  
- Experiments can be reproduced by checking out specific Git commits and DVC versions  

---

## Technologies Used
- Python (NumPy, Pandas, SciPy)
- Data visualization (Matplotlib, Seaborn, Plotly)
- Machine learning (Scikit-learn, XGBoost, LightGBM, CatBoost)
- Model explainability (SHAP)
- Data Version Control (DVC)
- Experiment tracking (MLflow)
- API framework (FastAPI)

---

## Intended Use
This repository is primarily intended for:
- Academic research and thesis work  
- Experimental analysis of crop suitability and yield prediction  
- Demonstration of reproducible machine learning workflows in agriculture  

The project may later be extended into a decision-support system or deployed application.

---

## Author
Lidetu Tadesse  
