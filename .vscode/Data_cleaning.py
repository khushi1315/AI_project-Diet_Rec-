import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
df= pd.read_csv("Datasets/Dataset.csv")
print(df.head())

df.info()

print(df.describe())

# Check for missing values
print(df.isnull().sum())
# Drop the 'Unnamed: 0' column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Remove rows with missing values
df.dropna(inplace=True)
# Remove duplicate rows
df.drop_duplicates(inplace=True)

#Fill missing values based on logic or using mean/median
df['age'].fillna(df['age'].median(), inplace=True)
df['weight(kg)'].fillna(df['weight(kg)'].median(), inplace=True)
df['height(m)'].fillna(df['height(m)'].median(), inplace=True)
# Calculate missing BMI if any
df['BMI'] = df['BMI'].fillna(df['weight(kg)'] / (df['height(m)'] ** 2))

# Fill missing BMR (Basal Metabolic Rate)
def calculate_bmr(row):
    if pd.isnull(row['BMR']):
        if row['gender'].lower() == 'male':
            return 88.362 + (13.397 * row['weight(kg)']) + (4.799 * row['height(m)'] * 100) - (5.677 * row['age'])
        else:
            return 447.593 + (9.247 * row['weight(kg)']) + (3.098 * row['height(m)'] * 100) - (4.330 * row['age'])
    return row['BMR']

df['BMR'] = df.apply(calculate_bmr, axis=1)
# Fill missing 'activity_level' and 'calories_to_maintain_weight' based on activity level
df['activity_level'].fillna('Sedentary', inplace=True)  # Default to Sedentary if unknown

activity_factors = {
    'Sedentary': 1.2,
    'Lightly Active': 1.375,
    'Moderately Active': 1.55,
    'Very Active': 1.725,
    'Extra Active': 1.9
}

df['calories_to_maintain_weight'] = df.apply(
    lambda x: x['bmr'] * activity_factors.get(x['activity_level'], 1.2) 
    if pd.isnull(x['calories_to_maintain_weight']) 
    else x['calories_to_maintain_weight'], 
    axis=1
)
    



#Remove Outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for numerical columns
df = remove_outliers(df, 'age')
df = remove_outliers(df, 'weight(kg)')
df = remove_outliers(df, 'height(m)')
df = remove_outliers(df, 'BMI')
df = remove_outliers(df, 'BMR')
df = remove_outliers(df, 'calories_to_maintain_weight')


# Create BMI tags if missing
def bmi_tag(BMI):
    if BMI < 18.5:
        return 'Underweight'
    elif 18.5 <= BMI < 24.9:
        return 'Normal weight'
    elif 25 <= BMI < 29.9:
        return 'Overweight'
    else:
        return 'Obesity'

df['BMI_tags'] = df['BMI'].apply(bmi_tag)

# Example label encoding for gender
df['gender'] = df['gender'].apply(lambda x: 1 if x.lower() == 'male' else 0)

# Normalize the weight and height for better model performance (Optional)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['weight(kg)', 'height(m)']] = scaler.fit_transform(df[['weight(kg)', 'height(m)']])
# Drop any remaining rows with NaN values (if any exist)
df.dropna(inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Save the cleaned dataset to a new Excel file
df.to_csv('cleaned_diet_recommendation_data.csv', index=False)

# Display the cleaned dataset info
print(df.info())
print(df.head())

#Adding new features

# 1. Add 'diet_type'
df['diet_type'] = np.random.choice(['Vegetarian', 'Vegan', 'Non-Vegetarian', 'Keto'], size=len(df))

# 2. Add 'health_conditions'
df['health_conditions'] = np.random.choice(['None', 'Diabetes', 'Hypertension', 'Heart Disease'], size=len(df))

# 3. Add 'goal'
df['goal'] = np.random.choice(['Weight Loss', 'Weight Gain', 'Muscle Gain', 'Maintenance'], size=len(df))
# 4. Add 'target_weight(kg)'
# Set target weight based on user goals
def calculate_target_weight(row):
    if row['goal'] == 'Weight Loss':
        return max(row['weight(kg)'] - np.random.uniform(5, 15), 40)
    elif row['goal'] == 'Weight Gain':
        return row['weight(kg)'] + np.random.uniform(5, 15)
    elif row['goal'] == 'Muscle Gain':
        return row['weight(kg)'] + np.random.uniform(3, 10)
    else:  # Maintenance
        return row['weight(kg)']

df['target_weight(kg)'] = df.apply(calculate_target_weight, axis=1)

# 5. Add 'steps_per_day'
# Randomly generate daily steps as a measure of physical activity
df['steps_per_day'] = np.random.randint(1000, 15000, size=len(df))

# 6. Add 'cholesterol_level'
df['cholesterol_level'] = np.random.choice(['Normal', 'High', 'Low'], size=len(df))

# 7. Add 'blood_pressure_level'
df['blood_pressure_level'] = np.random.choice(['Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2'], size=len(df))

# 8. Calculate a new column 'BMI_category' for better classification based on 'BMI'
def bmi_category(BMI):
    if BMI < 18.5:
        return 'Underweight'
    elif 18.5 <= BMI < 24.9:
        return 'Normal weight'
    elif 25 <= BMI < 29.9:
        return 'Overweight'
    else:
        return 'Obesity'

df['BMI_category'] = df['BMI'].apply(bmi_category)
# Save the expanded dataset as a CSV file
output_file = 'expanded_diet_recommendation_dataset.csv'
df.to_csv(output_file, index=False)

print(f"Dataset expanded with new features and saved as {output_file}.")

