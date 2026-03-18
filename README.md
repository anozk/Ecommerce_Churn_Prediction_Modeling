# E commerce churn prediction

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anozk/Ecommerce_Churn_Prediction_Modeling/blob/main/E_Commerce_Churn.ipynb)

## Overview
**Data:** [E-Commerce Customer Churn Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) — Kaggle  
**Part 2 — MLOps & AWS Deployment:** [Ecommerce Churn Model Deployment on AWS](https://github.com/anozk/Part-2-ML-Model-Deployment-on-AWS-for-Customer-Churn-Prediction)

**Keywords:** Imputation · Multicollinearity · Local Outlier Factor · Model Testing · Partial Dependence Plots · Ablation Study · Bootstrap Resampling · XGBoost · SHAP · Calibration

# Business Case: 
The E-commerce churn project classifies potential churners, meaning customers who want to leave the E-commerce website. The aim of the project is to test several machine learning models and to select the most suitable to classify as many churners as possible. This gives the E-commerce company the possibility to identify potential churners before they leave by offering discount coupons for instance. 

# Business Recommendation
Based on the SHAP analysis, two critical features emerged as key drivers for reducing churn: Cashback Amount and Warehouse-to-Home Delivery.
The SHAP plot reveals that higher Cashback Amounts (indicated by the blue area on the right side of the zero line) are associated with a significantly lower probability of churn. Conversely, lower cashback amounts (red area) correspond to an increased likelihood of customers leaving.
Additionally, the data suggests the company should prioritize improving its supply chain to reduce Warehouse-to-Home delivery times. This is evidenced by the red area in the plot, where more delivery days directly correlate with a higher probability of churn.

<img width="777" height="334" alt="download (5)" src="https://github.com/user-attachments/assets/4c5cf5e0-8763-4154-b9fe-b461db39ce58" />

# Methodology and approach: 

## Missing Values Imputation: Applying linear models, such as Logistic Regression, to data with improperly handled missingness can lead to biased odds ratio (OR) estimates, directly skewing final probability predictions. With several numeric features missing 3% to 8% of their values and a lack of specific domain metadata, both simple and iterative imputation strategies were evaluated. Comparative analysis demonstrated that Iterative Imputation superiorly preserved original data distributions and feature correlations, providing a more robust foundation for the classification model than simple univariate methods.
<img width="1482" height="983" alt="download" src="https://github.com/user-attachments/assets/0591e3db-abc8-4f0f-b2b3-1079ea9301b1" />

Multicollinearity Analysis: In addition to the three baseline models, \(k\)-Nearest Neighbors (kNN) and Logistic Regression were evaluated. While Logistic Regression is a linear model sensitive to multicollinearity in its coefficients, kNN—though non-linear—can also be negatively impacted if redundant features distort distance calculations. The initial data showed strong correlations among independent variables. To address this, one categorical variable was first removed to avoid the "dummy variable trap." Subsequently, an iterative feature selection process was performed using the Variance Inflation Factor (VIF = \(1/(1-R^{2})\)). Features with high VIF values were removed iteratively until all remaining variables showed acceptable levels of independence, specifically maintaining a VIF below 5. This same feature selection methodology was applied consistently across the training, validation, and test sets to maintain data integrity. 

Local Outlier Factor: To ensure a robust evaluation of \(k\)-Nearest Neighbors (kNN) and Logistic Regression, outliers were identified using the Local Outlier Factor (LOF) algorithm. First, numeric features were scaled using a standard scaler to ensure distance-based consistency. LOF was then applied using \(k=19\) neighbors. To prevent data leakage, the novelty=True parameter was utilized, and the decision_function was applied separately across the training, validation, and test sets. Observations predicted as outliers (labeled as -1) were removed from all three datasets. This preprocessing step is critical for linear models, as extreme values can disproportionately bias predictors and lead to misleading performance metrics. 
<img width="1584" height="784" alt="download (1)" src="https://github.com/user-attachments/assets/a724aa66-9cf8-4cc2-820b-1949b51c5b5e" />


Model Testing via Pipelines: We evaluated several algorithms, including K-Nearest Neighbors (KNN), Logistic Regression (LR), Random Forest (RF), LightGBM (LGBM), XGBoost (XGB), and Extra Trees (ET). Each was tested using initial parameters optimized for Recall and F1-score. We compared performance across three dataset variations: 1) Imputed data, 2) Imputed data with outliers removed, and 3) The raw training set (containing missing values and outliers). XGBoost achieved the highest Recall and F1-score on the raw dataset, leading to the decision to move forward with XGBoost as the primary model.

Partial Dependence Plots (PDP): After using Random Search to optimize the XGBoost parameters, the initial validation metrics for the churn class were exceptionally high (Precision: 0.94, Recall: 0.89, F1-score: 0.92). Partial Dependence Plots identified two dominant predictors: Complain and Tenure. A shift from a non-complaining to a complaining customer increases churn probability by 25 percentage points. Furthermore, a customer with less than one month of tenure is 45–50 percentage points more likely to churn than a customer with over 20 months of tenure.
Analysis revealed that including Tenure forced a strictly linear relationship on the numeric features, which failed to capture more complex churn behaviors. To avoid over-reliance on these dominant features and to address skepticism regarding the high accuracy values in a highly imbalanced dataset, Complain and Tenure were removed. This allowed the model to utilize the remaining features effectively and uncover less linear, more nuanced relationships.

<img width="482" height="2782" alt="download (2)" src="https://github.com/user-attachments/assets/ef3ed65d-ccdd-485e-949a-d75a430653b5" />


Ablation Study: An ablation study was conducted based on Leave-One-Feature-Out (LOFO) Importance to refine the feature set. First, a baseline F1 score was calculated for class 1. The study then iteratively removed one feature at a time, calculating the resulting delta (difference) in the F1 score.
Eleven features were identified where the delta increased regarding the F1 score, indicating they were harmful or redundant. These eleven features were excluded from the training, test, and validation sets, resulting in a final set of five features. Applying a subsequent random search with only these five features did not change the best hyperparameters.
When comparing the performance of the reduced feature set to the original non-reduced training set:

Reduced Set Metrics: Precision: 0.74, Recall: 0.81, F1-score: 0.77
Original Set Metrics: Precision: 0.85, Recall: 0.72, F1-score: 0.78

The results show a significant 9% increase in recall for the reduced model, while the overall F1 score only decreased slightly, pointing to a more robust and efficient final model. 

Calibration: The model was calibrated using the sigmoid method (Platt scaling) on the validation data, utilizing a 'prefit' strategy to ensure a proper separation between training and calibration. This process resulted in a 3.6% improvement in the Brier Score compared to the uncalibrated model.
This improvement is critical because a well-calibrated model ensures that predicted probabilities reflect reality: if the model predicts a 70% chance of churn, those customers actually leave approximately 70% of the time. Without calibration, models—especially tree-based ensembles—tend to be overconfident, often predicting extreme probabilities even when the actual risk is lower. By reducing the gap between predicted scores and actual outcomes, we have significantly boosted the trustworthiness and reliability of the 
model's output.

<img width="691" height="547" alt="download (3)" src="https://github.com/user-attachments/assets/5bdcb3ba-2e79-4de2-b9a6-2ede52121590" />


Bootstrap - Resampling: To avoid relying solely on a single point prediction on the test set, bootstrap resampling was applied. I generated 1,000 bootstrap samples to calculate 95% confidence intervals based on the precision-recall curve. Because bootstrapping uses sampling with replacement, it serves as a "stress test" for the model, creating data combinations that include more challenging edge cases and missing values (NaNs).
When comparing the calibrated model to the raw model, the 95% confidence intervals were slightly narrower, indicating improved stability. For the final test set prediction (using a 0.38 threshold), the model achieved a precision of 0.69, a recall of 0.72, and an F1-score of 0.70. This means 72 out of 100 actual churners were correctly identified (with 28 false negatives), while 69 out of 100 predicted churners actually left (with 31 false positives).

Confidence Intervals from bootstrap - resampling where recall is 0.72:

- Lower 95% CI Precision: 0.3541
- Median Precision:       0.4290
- Upper 95% CI Precision: 0.5076
  
Notably, the test precision of 0.69 falls outside the 95% confidence interval (Lower: 0.35, Median: 0.43, Upper: 0.51). This suggests that the bootstrap resampling is significantly more conservative than the test set, likely because the test set contains fewer "difficult" edge cases than the generated bootstrap samples. This outcome reinforces the importance of using bootstrap distributions over single point predictions to understand model performance limits.

<img width="1233" height="701" alt="download (4)" src="https://github.com/user-attachments/assets/7dd2cfe2-5071-4216-88e6-28af79b0dda9" />


