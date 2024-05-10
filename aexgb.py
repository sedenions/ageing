import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# Load the original methylation data
original_data_path = 'GTEx_Muscle.meth.csv'
original_data = pd.read_csv(original_data_path, sep='\t').set_index('cgID').transpose()

# Load the annotation data
annotation_data_path = 'GTEx_Muscle.anno.csv'
annotation_data = pd.read_csv(annotation_data_path, sep='\t')

# Extract age mapping from annotation data
age_mapping_series = annotation_data.iloc[9, 1:]
age_mapping = {key: int(value.split('-')[0]) for key, value in age_mapping_series.items()}

# Add the age group as a new column using the mapping
original_data['age_group'] = original_data.index.map(age_mapping).sort_values()

# Prepare the input features and target variable
X = original_data.drop('age_group', axis=1)  # Input features (methylation data)
y = original_data['age_group']  # Target variable (age group)

# Calculate the correlation coefficients
corr_coeffs = X.corrwith(y)

# Get the absolute values of the correlation coefficients
abs_corr_coeffs = np.abs(corr_coeffs)

# Select the top N features based on correlation coefficients
N = 1000  # Number of top features to select
top_features = abs_corr_coeffs.nlargest(N).index

# Create a new DataFrame with the selected features
X_selected = X[top_features]

# ENR Model
print("ENR Results:")
# Define the parameter grid for GridSearchCV
param_grid_enr = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9],
    'max_iter': [1000, 5000, 10000]
}

# Create an instance of the Elastic Net regressor
enet = ElasticNet(random_state=42)

# Perform grid search with cross-validation
grid_search_enr = GridSearchCV(estimator=enet, param_grid=param_grid_enr, cv=5, scoring='neg_mean_squared_error')
grid_search_enr.fit(X_selected, y)

# Get the best hyperparameters and model
best_params_enr = grid_search_enr.best_params_
best_model_enr = grid_search_enr.best_estimator_

print("Best hyperparameters: ", best_params_enr)

# Perform cross-validation with the best model
cv_scores_enr = cross_val_score(best_model_enr, X_selected, y, cv=5, scoring='neg_mean_squared_error')
cv_scores_enr = -cv_scores_enr  # Convert negative scores to positive
cv_mean_score_enr = cv_scores_enr.mean()
cv_std_score_enr = cv_scores_enr.std()

print("Cross-validation scores:", cv_scores_enr)
print("Mean cross-validation score:", cv_mean_score_enr)
print("Standard deviation of cross-validation scores:", cv_std_score_enr)

# Evaluate the best model on the test set
X_train_enr, X_test_enr, y_train_enr, y_test_enr = train_test_split(X_selected, y, test_size=0.2, random_state=42)
best_model_enr.fit(X_train_enr, y_train_enr)
y_pred_enr = best_model_enr.predict(X_test_enr)
mse_enr = mean_squared_error(y_test_enr, y_pred_enr)
mae_enr = mean_absolute_error(y_test_enr, y_pred_enr)
r2_enr = r2_score(y_test_enr, y_pred_enr)
print("Test set performance:")
print("Mean Squared Error:", mse_enr)
print("Mean Absolute Error:", mae_enr)
print("R-squared:", r2_enr)

# XGBoost Model
print("\nXGBoost Results:")
# Define the parameter grid for GridSearchCV
param_grid_xgb = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Create an instance of the XGBoost regressor
xgb = XGBRegressor(random_state=42)

# Perform grid search with cross-validation
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_selected, y)

# Get the best hyperparameters and model
best_params_xgb = grid_search_xgb.best_params_
best_model_xgb = grid_search_xgb.best_estimator_

print("Best hyperparameters: ", best_params_xgb)

# Perform cross-validation with the best model
cv_scores_xgb = cross_val_score(best_model_xgb, X_selected, y, cv=5, scoring='neg_mean_squared_error')
cv_scores_xgb = -cv_scores_xgb  # Convert negative scores to positive
cv_mean_score_xgb = cv_scores_xgb.mean()
cv_std_score_xgb = cv_scores_xgb.std()

print("Cross-validation scores:", cv_scores_xgb)
print("Mean cross-validation score:", cv_mean_score_xgb)
print("Standard deviation of cross-validation scores:", cv_std_score_xgb)

# Evaluate the best model on the test set
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_selected, y, test_size=0.2, random_state=42)
best_model_xgb.fit(X_train_xgb, y_train_xgb)
y_pred_xgb = best_model_xgb.predict(X_test_xgb)
mse_xgb = mean_squared_error(y_test_xgb, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test_xgb, y_pred_xgb)
r2_xgb = r2_score(y_test_xgb, y_pred_xgb)
print("Test set performance:")
print("Mean Squared Error:", mse_xgb)
print("Mean Absolute Error:", mae_xgb)
print("R-squared:", r2_xgb)