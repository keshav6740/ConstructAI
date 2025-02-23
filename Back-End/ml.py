import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

class ConstructionPredictor:
    def __init__(self):
        self.delay_model = None
        self.cost_model = None
        self.scaler = None
        self.feature_columns = [
            'square_footage', 'num_floors', 'soil_condition_score',
            'complexity_score', 'initial_estimated_duration',
            'initial_estimated_cost', 'avg_daily_workers',
            'equipment_utilization_rate'
        ]
        
    def preprocess_data(self, df):
        # Create copy of feature columns
        X = df[self.feature_columns].copy()
        
        # Create target variables
        y_delay = df['delay_percentage']
        y_cost = df['cost_overrun_percentage']
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_delay, y_cost
    
    def train_models(self, df):
        # Preprocess data
        X_scaled, y_delay, y_cost = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_delay_train, y_delay_test, y_cost_train, y_cost_test = train_test_split(
            X_scaled, y_delay, y_cost, test_size=0.2, random_state=42
        )
        
        # Initialize models
        self.delay_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.cost_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Train models
        self.delay_model.fit(X_train, y_delay_train)
        self.cost_model.fit(X_train, y_cost_train)
        
        # Evaluate models
        delay_pred = self.delay_model.predict(X_test)
        cost_pred = self.cost_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'delay_model': {
                'r2': r2_score(y_delay_test, delay_pred),
                'mse': mean_squared_error(y_delay_test, delay_pred),
                'mae': mean_absolute_error(y_delay_test, delay_pred)
            },
            'cost_model': {
                'r2': r2_score(y_cost_test, cost_pred),
                'mse': mean_squared_error(y_cost_test, cost_pred),
                'mae': mean_absolute_error(y_cost_test, cost_pred)
            }
        }
        
        return metrics
    
    def predict(self, project_data):
        # Ensure data is in correct format
        if isinstance(project_data, dict):
            project_data = pd.DataFrame([project_data])
        
        # Extract features
        X = project_data[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        delay_pred = self.delay_model.predict(X_scaled)
        cost_pred = self.cost_model.predict(X_scaled)
        
        return {
            'predicted_delay_percentage': delay_pred[0],
            'predicted_cost_overrun_percentage': cost_pred[0]
        }
    
    def save_models(self, path='construction_models'):
        """Save the trained models and scaler"""
        joblib.dump(self.delay_model, f'{path}_delay.joblib')
        joblib.dump(self.cost_model, f'{path}_cost.joblib')
        joblib.dump(self.scaler, f'{path}_scaler.joblib')
    
    def load_models(self, path='construction_models'):
        """Load the trained models and scaler"""
        self.delay_model = joblib.load(f'{path}_delay.joblib')
        self.cost_model = joblib.load(f'{path}_cost.joblib')
        self.scaler = joblib.load(f'{path}_scaler.joblib')

# Load the dataset
df = pd.read_csv('construction_projects_dataset.csv')

# Initialize and train the predictor
predictor = ConstructionPredictor()
metrics = predictor.train_models(df)

# Print model performance metrics
print("\nModel Performance Metrics:")
print("\nDelay Model:")
print(f"R² Score: {metrics['delay_model']['r2']:.3f}")
print(f"Mean Squared Error: {metrics['delay_model']['mse']:.3f}")
print(f"Mean Absolute Error: {metrics['delay_model']['mae']:.3f}")

print("\nCost Model:")
print(f"R² Score: {metrics['cost_model']['r2']:.3f}")
print(f"Mean Squared Error: {metrics['cost_model']['mse']:.3f}")
print(f"Mean Absolute Error: {metrics['cost_model']['mae']:.3f}")

# Example prediction
example_project = {
    'square_footage': 50000,
    'num_floors': 10,
    'soil_condition_score': 7.5,
    'complexity_score': 6.0,
    'initial_estimated_duration': 365,
    'initial_estimated_cost': 5000000,
    'avg_daily_workers': 100,
    'equipment_utilization_rate': 0.8
}

prediction = predictor.predict(example_project)
print("\nExample Prediction:")
print(f"Predicted Delay Percentage: {prediction['predicted_delay_percentage']:.2f}%")
print(f"Predicted Cost Overrun Percentage: {prediction['predicted_cost_overrun_percentage']:.2f}%")

# Save the models
predictor.save_models()