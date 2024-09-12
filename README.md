Employee Churn Prediction Project - README
Project Overview
This project aims to predict employee churn using various machine learning and deep learning models. It includes data preprocessing, exploratory data analysis (EDA), cluster analysis, model development, and model comparison. The project was deployed as a web application using Streamlit for ease of interaction with the prediction models.

Workflow
Data Cleaning

Handling missing values, outliers, and data inconsistencies.
Conversion of categorical columns to appropriate formats.
Exploratory Data Analysis (EDA)

Understanding the distribution of features (categorical and numerical) using visualizations such as histograms, count plots, and heatmaps.
Identifying correlations between features.
Cluster Analysis

Implemented cluster analysis to identify patterns within the dataset.
Used clustering algorithms to label customer behavior based on purchasing patterns and income.
Data Preprocessing

Applied encoding techniques:
OrdinalEncoder for ordinal categorical features like salary, work_accident and promotion_last_5years.
OneHotEncoder for nominal categorical features like departments.
Feature scaling for numerical data.
Splitting the dataset into training and testing sets.
Model Development

Trained multiple models to predict employee churn:
Logistic Regression
KNN Classifier
Random Forest
XGBoost
Deep Learning Model (Sequential)
Model Evaluation

Evaluated models using metrics such as F1 Score, Recall Score, and Average Precision Score.
Compared the models based on their performance metrics and accuracy.
Deployment

The project was deployed using Streamlit, providing an interactive web interface for users to:
Input employee data and receive predictions about their likelihood of leaving.
View prediction results with corresponding probabilities.
Compare model predictions by displaying results from different models.
How to Run the Project
1. Clone the repository:
   ![image](https://github.com/user-attachments/assets/ea4359bd-ca85-41e5-b4cc-e84c0265efca)

2. Install dependencies:
   ![image](https://github.com/user-attachments/assets/eca7b8a9-13f9-4536-b065-7b79983f5494)
   
3. Run the Streamlit application:
   ![image](https://github.com/user-attachments/assets/4433c846-a2b1-41ad-b746-b80e9b06b6ff)

4. The application will open in your browser, allowing you to interact with the prediction models and view results.

Project Structure
app.py: The main Streamlit application script.
data/: Directory containing the dataset.
models/: Directory with saved models for comparison.
notebooks/: Jupyter notebooks with EDA, clustering, and model training processes.
requirements.txt: Required Python packages for running the project.
README.md: This file.
Future Improvements
Adding more sophisticated feature engineering techniques.
Fine-tuning deep learning models.
Experimenting with additional algorithms for better accuracy.
