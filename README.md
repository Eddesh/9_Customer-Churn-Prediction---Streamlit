# Customer-Churn-Prediction---Streamlit

## Project Overview
This project involves building a churn prediction model for a bank's customer base. As a Data Scientist, my role is to develop a classification model that can predict customer churn effectively. The project follows a structured process, including data preprocessing, model training, model evaluation, and deployment. Two machine learning algorithms, Random Forest and XGBoost, were compared to find the optimal solution, which is then saved and deployed for future predictions.

### Dataset Description
The dataset contains information on bank customers, including various demographic, financial, and behavioral attributes that may influence their likelihood of churning. The columns in the dataset are as follows:

id: Unique identifier for each record.
CustomerId: Unique identifier for each customer.
Surname: Customer’s surname.
CreditScore: Customer's credit score.
Geography: Customer's country.
Gender: Customer's gender.
Age: Customer's age.
Tenure: Number of years the customer has been with the bank.
Balance: Customer's account balance.
NumOfProducts: Number of products the customer has with the bank.
HasCrCard: Indicates if the customer has a credit card (1 = Yes, 0 = No).
IsActiveMember: Indicates if the customer is an active member (1 = Yes, 0 = No).
EstimatedSalary: Customer's estimated salary.
Churn: Target variable indicating if the customer has churned (1 = Yes, 0 = No).

### Step 1: Data Preprocessing and Model Training
The Jupyter notebook file Modelling_1.ipynb contains the data preprocessing steps and the code for training and comparing two classification algorithms: Random Forest and XGBoost. The model with the highest performance is selected and saved as a pickle file for later use. 

### Step 2: Object-Oriented Programming (OOP) Model Integration
The OOP_2.py file is structured in an object-oriented approach, encapsulating the training process into classes and methods. This approach ensures that the training and model-saving processes are modular and reusable.

### Step 3: Prediction Code
Prediction_3.py provides a prediction code that utilizes the trained model to make predictions on new customer data. This code is designed for seamless integration with the deployment pipeline.

### Step 4: Deployment with Streamlit
The streamlit.ipynb file contains the deployment setup using Streamlit, allowing users to interact with the churn prediction model through a web interface. Two test cases are provided to demonstrate the model's functionality in real-time.

## Project Structure
├── Modelling_1.ipynb    # Data preprocessing and initial model training

├── OOP_2.py             # Object-oriented programming model integration

├── Prediction_3.py      # Prediction code for deployment

├── streamlit.ipynb      # Deployment using Streamlit

└── README.md            # Project documentation

## Author
Davin Edbert Santoso Halim
