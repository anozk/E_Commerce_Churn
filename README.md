# E coomerce churn prediction

# Business Case: 
The E-commerce churn project classifies potential churners, meaning customers who want to leave the E-commerce website. The aim of the project is to test several machiene learning models and to select the most suitable to classify as many churners as possible. This gives the E-commerce company the possibility to identify potential churners before they leave by offering discount coupons for instance. 

# Methodology and approach: 

Missing Values Imputation:Applying linear models, such as Logistic Regression, to data with improperly handled missingness can lead to biased odds ratio (OR) estimates, directly skewing final probability predictions. With several numeric features missing 3% to 8% of their values and a lack of specific domain metadata, both simple and iterative imputation strategies were evaluated. Comparative analysis demonstrated that Iterative Imputation superiorly preserved original data distributions and feature correlations, providing a more robust foundation for the classification model than simple univariate methods.

Multicollinearity Analysis:In addition to the three baseline models, \(k\)-Nearest Neighbors (kNN) and Logistic Regression were evaluated. While Logistic Regression is a linear model sensitive to multicollinearity in its coefficients, kNN—though non-linear—can also be negatively impacted if redundant features distort distance calculations. The initial data showed strong correlations among independent variables. To address this, one categorical variable was first removed to avoid the "dummy variable trap." Subsequently, an iterative feature selection process was performed using the Variance Inflation Factor (VIF = \(1/(1-R^{2})\)). Features with high VIF values were removed iteratively until all remaining variables showed acceptable levels of independence, specifically maintaining a VIF below 5. This same feature selection methodology was applied consistently across the training, validation, and test sets to maintain data integrity. 

To ensure a robust evaluation of \(k\)-Nearest Neighbors (kNN) and Logistic Regression, outliers were identified using the Local Outlier Factor (LOF) algorithm. First, numeric features were scaled using a standard scaler to ensure distance-based consistency. LOF was then applied using \(k=19\) neighbors. To prevent data leakage, the novelty=True parameter was utilized, and the decision_function was applied separately across the training, validation, and test sets. Observations predicted as outliers (labeled as -1) were removed from all three datasets. This preprocessing step is critical for linear models, as extreme values can disproportionately bias predictors and lead to misleading performance metrics. 

Model Testing via Pipelines: We evaluated several algorithms, including K-Nearest Neighbors (KNN), Logistic Regression (LR), Random Forest (RF), LightGBM (LGBM), XGBoost (XGB), and Extra Trees (ET). Each was tested using initial parameters optimized for Recall and F1-score. We compared performance across three dataset variations: 1) Imputed data, 2) Imputed data with outliers removed, and 3) The raw training set (containing missing values and outliers). XGBoost achieved the highest Recall and F1-score on the raw dataset, leading to the decision to move forward with XGBoost as the primary model.

Partial Dependence Plots (PDP): After using Random Search to optimize the XGBoost parameters, the initial validation metrics for the churn class were exceptionally high (Precision: 0.94, Recall: 0.89, F1-score: 0.92). Partial Dependence Plots identified two dominant predictors: Complain and Tenure. A shift from a non-complaining to a complaining customer increases churn probability by 25 percentage points. Furthermore, a customer with less than one month of tenure is 45–50 percentage points more likely to churn than a customer with over 20 months of tenure.
Analysis revealed that including Tenure forced a strictly linear relationship on the numeric features, which failed to capture more complex churn behaviors. To avoid over-reliance on these dominant features and to address skepticism regarding the high accuracy values in a highly imbalanced dataset, Complain and Tenure were removed. This allowed the model to utilize the remaining features effectively and uncover less linear, more nuanced relationships.


Ablation Study: An ablation study was conducted based on Leave-One-Feature-Out (LOFO) Importance to refine the feature set. First, a baseline F1 score was calculated for class 1. The study then iteratively removed one feature at a time, calculating the resulting delta (difference) in the F1 score.
Eleven features were identified where the delta increased regarding the F1 score, indicating they were harmful or redundant. These eleven features were excluded from the training, test, and validation sets, resulting in a final set of five features. Applying a subsequent random search with only these five features did not change the best hyperparameters.
When comparing the performance of the reduced feature set to the original non-reduced training set:

Reduced Set Metrics: Precision: 0.74, Recall: 0.81, F1-score: 0.77
Original Set Metrics: Precision: 0.85, Recall: 0.72, F1-score: 0.78

The results show a significant 9% increase in recall for the reduced model, while the overall F1 score only decreased slightly, pointing to a more robust and efficient final model. 

Calibration: The model was calibrated using the sigmoid method (Platt scaling) on the validation data, utilizing a 'prefit' strategy to ensure a proper separation between training and calibration. This process resulted in a 3.6% improvement in the Brier Score compared to the uncalibrated model.
This improvement is critical because a well-calibrated model ensures that predicted probabilities reflect reality: if the model predicts a 70% chance of churn, those customers actually leave approximately 70% of the time. Without calibration, models—especially tree-based ensembles—tend to be overconfident, often predicting extreme probabilities even when the actual risk is lower. By reducing the gap between predicted scores and actual outcomes, we have significantly boosted the trustworthiness and reliability of the model's output.

Bootstrap - Resampling: To avoid relying solely on a single point prediction on the test set, bootstrap resampling was applied. I generated 1,000 bootstrap samples to calculate 95% confidence intervals based on the precision-recall curve. Because bootstrapping uses sampling with replacement, it serves as a "stress test" for the model, creating data combinations that include more challenging edge cases and missing values (NaNs).
When comparing the calibrated model to the raw model, the 95% confidence intervals were slightly narrower, indicating improved stability. For the final test set prediction (using a 0.38 threshold), the model achieved a precision of 0.69, a recall of 0.72, and an F1-score of 0.70. This means 72 out of 100 actual churners were correctly identified (with 28 false negatives), while 69 out of 100 predicted churners actually left (with 31 false positives).

Confidence Intervals from bootstrap - resampling where recall is 0.72:

- Lower 95% CI Precision: 0.3541
- Median Precision:       0.4290
- Upper 95% CI Precision: 0.5076
  
Notably, the test precision of 0.69 falls outside the 95% confidence interval (Lower: 0.35, Median: 0.43, Upper: 0.51). This suggests that the bootstrap resampling is significantly more conservative than the test set, likely because the test set contains fewer "difficult" edge cases than the generated bootstrap samples. This outcome reinforces the importance of using bootstrap distributions over single point predictions to understand model performance limits.




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
