# Credit Wise: Loan Approval Prediction Using KNN

## Title of the Project

**Credit Wise - Loan Approval Prediction Using K-Nearest Neighbors (KNN)**

## Problem Statement

Loan approval decisions depend on several applicant, financial, and loan-related factors. Manual assessment can be time-consuming and inconsistent. This project develops a machine-learning model to predict whether a loan application will be approved.

## Objective of the Project

* Analyze loan application data and identify useful approval-related patterns.
* Prepare incomplete numeric and categorical data for machine learning.
* Engineer financial risk and repayment-capacity features.
* Train and tune a KNN classification model.
* Improve and evaluate the model using precision and F1 score.

## Real-life Application

Financial institutions can use this type of model as a decision-support tool for loan officers. It can help prioritize applications, identify applications requiring additional review, and make preliminary approval screening faster and more consistent. Final decisions should still include responsible human and regulatory review.

## Dataset Collection and Description

The project uses `loan_approval_data.csv`, located in the project directory. It contains 1,000 loan applications and includes:

* Applicant income and coapplicant income
* Employment, marital, education, gender, and employer information
* Age and number of dependents
* Credit score, existing loans, and debt-to-income ratio
* Savings, collateral value, loan amount, and loan term
* Loan purpose and property area
* `Loan_Approved`, the target variable (`Yes` or `No`)

The dataset includes missing values in several columns. Rows without a target label are excluded from model training.

## Data Preprocessing

* Removed rows with missing `Loan_Approved` values.
* Removed `Applicant_ID` because it is an identifier, not a predictive feature.
* Imputed missing numeric values using the median.
* Imputed missing categorical values using the most frequent category.
* Standardized numeric features.
* One-hot encoded categorical features.
* Used a pipeline so preprocessing is fitted only on the training data.

## Exploratory Data Analysis (EDA)

The notebook explores:

* Loan approval class distribution
* Education and other categorical variables
* Applicant and coapplicant income distributions
* Income, credit score, DTI ratio, and savings by approval outcome
* Correlations between numeric features and loan approval

## Implementation of K-Means or KNN Algorithm

This project implements **K-Nearest Neighbors (KNN)** classification. KNN predicts an application by comparing it with nearby training examples in the scaled feature space.

Logistic Regression and Naive Bayes were removed; KNN is the only final model.

Additional engineered features include:

* Total income
* Loan-to-income ratio
* Savings-to-loan ratio
* Collateral-to-loan ratio
* Income per dependent
* Credit-score/DTI interaction
* Squared credit score and DTI ratio

## Model Training

The data is split into training and test sets using a stratified split with `random_state=42`. KNN hyperparameters are tuned using five-fold stratified cross-validation:

* Number of neighbors
* Distance weighting
* Manhattan or Euclidean distance
* Leaf size

The best configuration is selected using cross-validated F1 score.

## Model Evaluation

The final model is evaluated using:

* Precision
* Recall
* F1 score
* Accuracy
* Confusion matrix

Precision and F1 score are emphasized because the project aims to reduce incorrect approval predictions while maintaining a balanced classification result.

## Results and Visualization

On the fixed stratified test split, the tuned KNN model achieved:

| Metric | Score |
|---|---:|
| Precision | 0.894 |
| Recall | 0.700 |
| F1 score | 0.785 |
| Accuracy | 0.879 |

The notebook includes pie charts, histograms, box plots, a correlation heatmap, and a confusion matrix to visualize the data and model performance.

## Conclusion

The engineered features and cross-validated KNN tuning improved the model over the original fixed-parameter KNN approach. The final model provides strong precision and a substantially improved F1 score for loan approval prediction. It should be treated as a decision-support system and validated further with larger, current, and representative lending data before production use.

## Project Files

* `credit_wise.ipynb` - complete analysis, training, and evaluation workflow.
* `loan_approval_data.csv` - loan application dataset.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook credit_wise.ipynb
```

Run the notebook cells from top to bottom with `loan_approval_data.csv` in the same directory.

## Interactive Prediction UI

The project also includes a Streamlit interface where users can enter applicant and loan details and receive a preliminary approval prediction.

Start the application with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal. The interface displays:

* A polished Credit Wise dashboard
* Applicant, employment, and financial input fields
* Predicted approval outcome
* Estimated approval likelihood
* A responsible-use notice explaining that the prediction is not a final lending decision
