# E coomerce churn prediction

# Business Case: 
The E-commerce churn project classifies potential churners, meaning customers who want to leave the E-commerce website. The aim of the project is to test several machiene learning models and to select the most suitable to classify as many churners as possible. This gives the E-commerce company the possibility to identify potential churners before they leave by offering discount coupons for instance. 

# Methodology: 

Missing Values Imputation: 

Multicollinearity: In the project apart from three based models, also knn-Neighbourhood and Logistic Regression are tested. Both are linear models meaning that multilicollinearity can have an impact on the prediction and bias the prediction.  
In the data there is a strong correlation among the independent variables. First, one of the categorical varialbes have been removed to avoid dummy trap (VIF = 1/1-R^2), then based on VIF value iteratively features have been selected and removed one by one. The same methodology has been applied to training, validation and test sets seperatly. 

Local Outlier Factor: To test correctly knn-Neighbourhood and Logistic Regression, outlier have been indentified apppliying local outliers factor. Firstly, the numeric feature have been scaled using standard scaler. Then local outlier factor has been applied to with using 19 neighbourhoods. To avoid data leakage clf_novelty.decision_function have been applied to trainin, validation and test sets. Predicted -1 by the algorithm are selected as outliers and have been deleted from the three data sets. Outlier are contraproductiv for lienar models because they can bias the predictor and lead to unaccurate accuracy metrics. 

Models tests through pipelines: Knn-Neighborhood, Losgistic Regression (LR), Random Forest (RF), Light Gradient Boosting Machine (lgb), XG Boost (XGB) and Extra Tree Model (ET) have been tested with a inintial parameter setting based on recall and f1 score. Several datasets have been tested: Imputed, the impupted and outlier removed and a trainng set that includes missing values, non imputed and no outlier removed. The highest recall and f1 score achieved XGB on the non imputred, non outlier removed with missing values. So the decision was to continue with XGB.

Partial dependence plots: Random search has been applied on XGB to find the best parameters. Initially accurcay metrics on the validation data for the first class were as follows: Precision, Recall and F1-score, 0.94, 0.89, 0.92 respectively. Two features have been identified as very strong predictors. Complain and Tenure: According to partial dependence plots, a change from non complaining to a complaining customer increases churn by 25 percentage points. Also, regarding tenure, A customer who is less than a month on the E-commerce site compared to a customer who more than 20 months on the site decreases the probability of churn by almst 45 - 50 percentage points. Also when analyzing the nueric feature with including tenure they have a linear relationship to churn not contributin analysing chrun behavoir. On the other hand excluding Tenure feature the other numeric feature have a higher impact on chrun less linear relationshp. The tenure and complain feature have been removed to utilize the other features and of the sceptistism regarding the high accurcy valeus in a highy imbalanced data set. 

Ablation study:Ablation was made based on Leave-One-Feature-Out (LOFO) Importance. First the study calculated baseline f1 score for calss 1. By leaving one feature out the f1 score the delta meaning the differene has been calculated. Eleven features have been identified where the delta increased regardin the f1 score. This eleven featrues habbe excluded from the trainin, test and validation sets, remaining five feautres. Applying random search with five features did not change best parameters. However, comparing the prediction of reduced features with the non reduce feautres training set, precision, recall and f1 score 0.74, 0.81, 0.77 with 0.85      0.72 0.78, respectively, recall increased by 9% while f1 score only decreased slightly. 

Calibration: The model has been calibrated based on the validation data applying sigmoid method and prefit for cross validation. Compared to non raw model there is a 3.6% improvement. Calibration is important because well-calibrated model ensures that when it predicts a 70% chance of churn, those customers actually leave approximately 70% of the time. Without calibration a model might be "overconfident," predicting 99% churn for many people, even if the actual risk is lower.
Calibration reduces the gap between the predicted scores and reality, boosting the trustworthiness of your results.

Bootstrap - Resampling: In order to no to just to do a punctual prediction on the test set, bootstrap resampling has been applied. 1000 bootstrap samples have been generated based on precision - recall curve with 95% confidence intervalls. In addition, the bootstrap is applied with replacement which is then a stress test for the model. There will be a combination of data sets that contains more difficult data points as well as NaN values. When comparing the calibrated model to the raw model the 95% confidence intervalls have been slightly narrower. For the final prediction on the test set the model achieved with a threshold of .38, 0.69, 0.72, 0.70 for precision, recall and f1 score, respectively. We can say that 72 out of 100 churners will be identifed as churners and there will be 28 false negative. On the other hand, 69 of 100 customers will churn and there will be 31 false positive. Comparing the final metric with the 95% confidence intervalls: At Recall 0.72:
- Lower 95% CI Precision: 0.3541
- Median Precision:       0.4290
- Upper 95% CI Precision: 0.5076
Achieving a precision of 0.69 is out side, on the upper part of the 95% confidence intervalls. The assumption is that the bootstrap - reasampling is much more conservative way of predicting the test set. 

1. Essential Structure for 2026
A professional data science README should include these core sections:
Project Title & Hook: A clear name followed by a one-line "elevator pitch" describing the value of the project.
Business Case/Motivation: Explain why you built this. What problem does it solve for an E-commerce company?.
Key Highlights (The "Selling Points"): Use a bulleted list to showcase your advanced techniques:
Bootstrap Resampling for robust performance metrics.
Ablation Study to identify critical features.
Model Calibration for reliable probability scores.
Data Description: Mention your data source (e.g., Kaggle, scraped data) and any critical preprocessing like your specific imputation strategy.
Tech Stack: List your primary tools using icons or a clear list (e.g., Python 3.12, Scikit-learn, XGBoost, Pandas).
Results & Evaluation: Don't just list code; show the "so what." Include your final accuracy, ROC-AUC, or calibration curves. 
2. Make it Visual (Crucial for Portfolios)
GitHub users often skim looking at pictures. You can add charts directly to your README: 
Upload the Image: Save your best plot (like a confusion matrix or calibration curve) from Colab as a .png and upload it to your GitHub repo.
Embed in Markdown: Use this syntax: ![Alt Text](path/to/your/image.png).
Use Badges: Add small, colorful status badges from Shields.io to show your project's tech stack or license status. 
3. Professional Polish Tips
Table of Contents: If your README is long, add a clickable table of contents at the top for easy navigation.
Reproducibility: Include a requirements.txt file and a section on how to run your code locally.
Future Work: Mention 1–2 things you would do next (e.g., "Implement Deep Learning" or "Deploy as a Streamlit App").
Tone: Keep it professional, concise, and free of typos. 
