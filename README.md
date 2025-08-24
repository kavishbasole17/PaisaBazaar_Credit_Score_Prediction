# PaisaBazaar Credit Score Prediction

A machine learning project to classify customer credit scores using Python and Scikit-learn. The final model, a Random Forest Classifier, provides a data-driven tool to automate and improve the accuracy of credit risk assessment.

---

## Problem Statement

The goal of this project is to address the challenge of manual and time-consuming credit risk assessment. By leveraging machine learning, we aim to build a robust model that can accurately predict a customer's credit score category ('Good', 'Standard', or 'Poor') based on their financial and personal information.

This predictive tool is intended to help PaisaBazaar:

* **Automate** the credit approval process.
* **Reduce** the time taken to make lending decisions.
* **Improve** the consistency and accuracy of credit scoring.
* **Minimize** financial risk by correctly identifying high-risk applicants.

---

## Dataset

The project uses the `Paisabazar.csv` dataset, which contains anonymized customer data. It includes 28 features representing a mix of personal information (e.g., `Age`, `Occupation`), financial history (e.g., `Annual_Income`, `Num_of_Loan`), and credit-related metrics.

---

## Project Workflow

The project follows a standard data science methodology:

1.  **Data Cleaning & Preprocessing:** Handled missing values, identified and removed high-cardinality features to prevent memory errors, and prepared the data for modeling.
2.  **Exploratory Data Analysis (EDA):** Used visualizations like countplots and heatmaps to understand feature distributions and relationships.
3.  **Hypothesis Testing:** Performed an ANOVA test to statistically validate the relationship between key numerical features (`Annual_Income`) and the target variable (`Credit_Score`).
4.  **Feature Engineering:** Applied Label Encoding to the target variable and One-Hot Encoding to categorical features. Used `StandardScaler` to scale numerical features.
5.  **Model Implementation:** Trained and evaluated three different classification models: Logistic Regression, Decision Tree, and Random Forest.
6.  **Model Evaluation:** Compared models based on Accuracy, Precision, Recall, and F1-Score to select the best-performing model.

---

## Model Performance

The performance of the trained models was compared to select the most effective one for this task.

| Model               | Accuracy |
| ------------------- | -------- |
| **Random Forest** | **~83.14%** |
| Decision Tree       | ~81.40%   |
| Logistic Regression | ~77.60%   |


The **Random Forest Classifier** was chosen as the final model due to its superior accuracy and robustness against overfitting.

---

## Feature Importance

One of the key advantages of the Random Forest model is its ability to calculate feature importance. This helps us understand which factors are the most influential in predicting a customer's credit score.

The analysis shows that financial metrics like `Annual_Income`, `Interest_Rate`, and `Num_of_Loan` are among the most significant predictors.

---

## Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [[https://github.com/your-username/your-repository-name.git](https://github.com/kavishbasole17/PaisaBazaar_Credit_Score_Predciction.git)](https://github.com/kavishbasole17/PaisaBazaar_Credit_Score_Predciction.git)
    cd PaisaBazaar_Credit_Score_Predciction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook PaisaBazaar_Credit_Score_Prediction.ipynb
    ```

---

## Technologies Used

* **Python**
* **Pandas** & **NumPy** for data manipulation
* **Matplotlib** & **Seaborn** for data visualization
* **Scikit-learn** for machine learning modeling and preprocessing

---

## Future Work

* **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to find the optimal parameters for the Random Forest model to further boost performance.
* **Advanced Models:** Experiment with gradient boosting libraries like **XGBoost** or **LightGBM**, which are industry standards for performance on tabular data.
* **Deployment:** Save the final model using `joblib` and build a simple API with **Flask** or **FastAPI** to serve real-time predictions.
