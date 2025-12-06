# machine-learning
This repository contains hands-on Machine Learning exercises using Python and scikit-learn.  It includes simple regression and classification models, train/test split examples,  model evaluation, and foundational ML workflows suitable for beginners.

📌 Project Overview

This module introduces:

What Machine Learning is

The difference between Regression and Classification

How to prepare data using train/test split

How ML models learn patterns from data

Evaluating model performance

Understanding overfitting vs underfitting

Running ML models locally on your machine

These examples are intentionally simple so that beginners can understand the ML workflow clearly.

📁 Repository Structure
python-machine-learning-basics/
│── regression_example.py              # Linear Regression example
│── classification_example.py          # Decision Tree Classification example
│── week2-ml-practice.ipynb            # Notebook version (optional)
│── README.md                          # Documentation

🧰 Technologies & Libraries
Component	Usage
Python 3.x	Core programming language
NumPy	Numerical operations
Pandas	Dataset loading & cleaning
scikit-learn	ML models & evaluation
Jupyter/Colab (optional)	Notebook environment
📦 Installation

Before running the scripts, install the dependencies.

1. Create a virtual environment (optional but recommended)
python -m venv venv

2. Activate it

Windows:

venv\Scripts\activate

3. Install required libraries
pip install pandas numpy scikit-learn

▶️ How to Run the Project
Run Linear Regression model
python regression_example.py

Run Decision Tree Classification model
python classification_example.py

Output includes:

Model predictions

Train/Test evaluation

Metrics such as:

MSE (Mean Squared Error) for Regression

Accuracy Score for Classification

📊 Example Outputs
Regression (House Price Prediction)
Predictions: [130.]
MSE: 0.0

Classification (Buy/Not Buy Prediction)
Predictions: [1]
Accuracy: 1.0


Note: Output may vary slightly depending on dataset split.
