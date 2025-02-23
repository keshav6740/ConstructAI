import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_construction_dataset(num_projects=1000):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Basic project information
    project_types = ['Residential', 'Commercial', 'Industrial', 'Infrastructure']
    construction_phases = ['Foundation', 'Structure', 'Interior', 'Exterior', 'MEP', 'Finishing']
    weather_conditions = ['Sunny', 'Rainy', 'Cloudy', 'Stormy', 'Snow']
    
    # Generate base data
    data = {
        'project_id': range(1, num_projects + 1),
        'project_type': [random.choice(project_types) for _ in range(num_projects)],
        'location': [f"Site_{i}" for i in range(num_projects)],
        'initial_estimated_duration': np.random.randint(90, 730, num_projects),  # 3 months to 2 years
        'initial_estimated_cost': np.random.uniform(500000, 10000000, num_projects),
        'square_footage': np.random.randint(1000, 100000, num_projects),
        'num_floors': np.random.randint(1, 50, num_projects),
        'soil_condition_score': np.random.uniform(0, 10, num_projects),
        'complexity_score': np.random.uniform(1, 10, num_projects)
    }
    
    # Calculate actual duration with some variance
    data['actual_duration'] = [
        int(dur * np.random.uniform(0.8, 1.4))  # Actual duration varies from estimate
        for dur in data['initial_estimated_duration']
    ]
    
    # Calculate actual cost with some variance
    data['actual_cost'] = [
        cost * np.random.uniform(0.9, 1.5)  # Actual cost varies from estimate
        for cost in data['initial_estimated_cost']
    ]
    
    # Generate weather-related delays
    data['weather_delays'] = np.random.randint(0, 30, num_projects)
    
    # Generate resource-related metrics
    data['avg_daily_workers'] = np.random.randint(10, 200, num_projects)
    data['equipment_utilization_rate'] = np.random.uniform(0.6, 0.95, num_projects)
    
    # Generate safety metrics
    data['safety_incidents'] = np.random.randint(0, 5, num_projects)
    data['safety_score'] = np.random.uniform(7, 10, num_projects)
    
    # Generate quality metrics
    data['defects_found'] = np.random.randint(0, 50, num_projects)
    data['quality_score'] = np.random.uniform(6, 10, num_projects)
    
    # Calculate delay and cost overrun
    data['delay_in_days'] = [
        actual - estimated 
        for actual, estimated in zip(data['actual_duration'], data['initial_estimated_duration'])
    ]
    
    data['cost_overrun'] = [
        actual - estimated 
        for actual, estimated in zip(data['actual_cost'], data['initial_estimated_cost'])
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add derived metrics
    df['cost_per_sqft'] = df['actual_cost'] / df['square_footage']
    df['productivity_rate'] = df['square_footage'] / (df['actual_duration'] * df['avg_daily_workers'])
    df['cost_overrun_percentage'] = (df['cost_overrun'] / df['initial_estimated_cost']) * 100
    df['delay_percentage'] = (df['delay_in_days'] / df['initial_estimated_duration']) * 100
    
    # Round numerical columns for readability
    for column in df.select_dtypes(include=[np.float64]).columns:
        df[column] = df[column].round(2)
    
    return df

# Generate the dataset
construction_df = generate_construction_dataset()

# Display first few rows and basic statistics
print("\nFirst few rows of the dataset:")
print(construction_df.head())

print("\nDataset Statistics:")
print(construction_df.describe())

# Save to CSV
construction_df.to_csv('construction_projects_dataset.csv', index=False)