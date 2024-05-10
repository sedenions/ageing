import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

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

# Select the top N features based on correlation coefficients
N = 1000  # Number of top features to select
corr_coeffs = X.corrwith(y)
abs_corr_coeffs = np.abs(corr_coeffs)
top_features = abs_corr_coeffs.nlargest(N).index
X_selected = X[top_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super(Net, self).__init__()
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Define the hyperparameter search space
param_grid = {
    'hidden_sizes': [
        [256, 128, 64, 32],
        [128, 64, 32],
        [64, 32],
    ],
    'lr': [0.001, 0.01],
    'batch_size': [32, 64],
}

# Create an instance of the neural network
input_dim = X_train_tensor.shape[1]
model = Net(input_dim, hidden_sizes=[])

# Define the loss function
criterion = nn.MSELoss()

# Perform hyperparameter search using GridSearchCV
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

def train_and_evaluate(model, X_train, y_train, X_test, y_test, criterion, lr, batch_size, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    with torch.no_grad():
        y_pred = model(X_test)
        mse = mean_squared_error(y_test, y_pred.numpy())
        mae = mean_absolute_error(y_test, y_pred.numpy())
        r2 = r2_score(y_test, y_pred.numpy())
    
    return mse, mae, r2

def grid_search(model, X_train, y_train, X_test, y_test, param_grid, criterion, num_epochs):
    best_params = None
    best_mse = float('inf')
    
    for hidden_sizes in param_grid['hidden_sizes']:
        for lr in param_grid['lr']:
            for batch_size in param_grid['batch_size']:
                layers = []
                prev_size = input_dim
                for hidden_size in hidden_sizes:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(nn.ReLU())
                    prev_size = hidden_size
                layers.append(nn.Linear(prev_size, 1))
                model.layers = nn.Sequential(*layers)
                
                mse, mae, r2 = train_and_evaluate(model, X_train, y_train, X_test, y_test, criterion, lr, batch_size, num_epochs)
                
                if mse < best_mse:
                    best_params = {
                        'hidden_sizes': hidden_sizes,
                        'lr': lr,
                        'batch_size': batch_size
                    }
                    best_mse = mse
    
    return best_params

# Perform hyperparameter search
best_params = grid_search(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, param_grid, criterion, num_epochs=100)

# Train the model with the best hyperparameters
layers = []
prev_size = input_dim
for hidden_size in best_params['hidden_sizes']:
    layers.append(nn.Linear(prev_size, hidden_size))
    layers.append(nn.ReLU())
    prev_size = hidden_size
layers.append(nn.Linear(prev_size, 1))
model.layers = nn.Sequential(*layers)

mse, mae, r2 = train_and_evaluate(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, criterion, best_params['lr'], best_params['batch_size'], num_epochs=100)


print("Best hyperparameters:", best_params)
print("Test set performance:")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)