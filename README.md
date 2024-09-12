# Employee Churn Prediction Project

## Project Overview
This project aims to predict employee churn using various machine learning and deep learning models. It includes data preprocessing, exploratory data analysis (EDA), cluster analysis, model development, and model comparison. The project was deployed as a web application using **Streamlit** for ease of interaction with the prediction models.

## Workflow

### Data Cleaning
- Handling missing values, outliers, and data inconsistencies.
- Conversion of categorical columns to appropriate formats.

### Exploratory Data Analysis (EDA)
- Understanding the distribution of features (categorical and numerical) using visualizations such as histograms, count plots, and heatmaps.
- Identifying correlations between features.

### Cluster Analysis
- Implemented cluster analysis to identify patterns within the dataset.
- Used clustering algorithms to label customer behavior based on purchasing patterns and income.

### Data Preprocessing
- Applied encoding techniques:
  - **OrdinalEncoder** for ordinal categorical features like `salary`, `work_accident`, and `promotion_last_5years`.
  - **OneHotEncoder** for nominal categorical features like `departments`.
- Feature scaling for numerical data.
- Splitting the dataset into training and testing sets.

### Model Development
Trained multiple models to predict employee churn:
- Logistic Regression
- KNN Classifier
- Random Forest
- XGBoost
- Deep Learning Model (Sequential)

### Model Evaluation
- Evaluated models using metrics such as **F1 Score**, **Recall Score**, and **Average Precision Score**.
- Compared the models based on their performance metrics and accuracy.

### Deployment
The project was deployed using **Streamlit**, providing an interactive web interface for users to:
- Input employee data and receive predictions about their likelihood of leaving.
- View prediction results with corresponding probabilities.
- Compare model predictions by displaying results from different models.

## How to Run the Project

## 1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd employee-churn-prediction
```
 ## 2. Install dependencies::

   ```bash
   pip install -r requirements.txt
 ```
## 3. Run the Streamlit application:
   ```bash
   streamlit run app.py
```
## 4. The application will open in your browser, allowing you to interact with the prediction models and view results.
### Project Structure
```bash
employee-churn-prediction/
├── app.py                   # Main Streamlit application script
├── data/                    # Directory containing the dataset
├── models/                  # Directory with saved models for comparison
├── notebooks/               # Jupyter notebooks with EDA, clustering, and model training processes
├── requirements.txt         # Required Python packages for running the project
└── README.md                # This file
```
### Visualization of Streamlit File
Below are screenshots from the Streamlit app:
![1](https://github.com/user-attachments/assets/880ac040-95bc-4c45-a9a0-1c336718d918)
(`---`)
![2](https://github.com/user-attachments/assets/23666f0d-00aa-45a4-8a3b-e2d663335d19)
(`---`)
![3](https://github.com/user-attachments/assets/f8e8a6a2-51b1-4550-a73c-928eed963b09)
(`---`)
![4](https://github.com/user-attachments/assets/8dd06ae0-d64a-4281-baf0-fd2353bc7a6a)
(`---`)
![5](https://github.com/user-attachments/assets/9feaaf90-f7cb-4853-95d4-3f3fbc709994)
(`---`)
![6](https://github.com/user-attachments/assets/c89eaa09-6194-414a-bb61-23c156461535)
(`---`)
![7](https://github.com/user-attachments/assets/31ce90b0-a82e-464b-b5f7-d51339ebbed1)
(`---`)


