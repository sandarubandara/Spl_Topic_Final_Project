from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('final_crop_analysis.csv')
print(df.head())
print('shape of the dataset is: ',df.shape)
print('\nGeneral information of the datsaset is: \n')
print(df.info())

#Since column 'Code' is not required for our analysis, we will drop the column
df.drop(columns='Code', inplace=True, errors='ignore')
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

df.duplicated().sum()

df['Year'] = df['Year'].astype('object')

# Select only numeric columns for correlation matrix
numeric_cols = df.select_dtypes(include=['float64', 'int64'])
categorical_cols = df.select_dtypes(include=['object'])

print(numeric_cols.columns)

print(categorical_cols.columns)
""" 
# Plot histograms for numerical columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 6, i + 1)
    sb.histplot(df[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Plot box plots for numerical columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 6, i + 1)
    sb.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Plot bar plots for categorical columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(2, 3, i + 1)
    sb.countplot(x=df[col])
    plt.title(col)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
 """
# Calculate and print skewness
for col in numeric_cols:
    skewness = stats.skew(df[col])
    print(f"Skewness of {col}: {skewness}")

# Loop through each categorical column and print value counts
for col in categorical_cols:
    print(f"Value counts for {col}:")
    print(df[col].value_counts())
    print("-" * 20) # Separator for clarity

# Check for outliers using IQR method
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"{col}: Found {len(outliers)} potential outliers")

skewed_cols = ['Economic_Impact_Million_Usd', 'Machinery_Per_Ag_Land', 'Ag_Land_Index', 'Labor_Index']

# Verify and process only existing numeric columns in skewed_cols
valid_skewed_cols = [col for col in skewed_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
df_before_capping = df.copy()
print(df_before_capping.columns)
for col in valid_skewed_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] > upper_bound, upper_bound, 
                       np.where(df[col] < lower_bound, lower_bound, df[col]))
    
for col in valid_skewed_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"{col}: Found {len(outliers)} potential outliers")

# Visualize before capping
""" plt.figure(figsize=(15, 10))
for i, col in enumerate(skewed_cols):
    plt.subplot(2, 2, i + 1)
    sb.boxplot(x=df_before_capping[col])
    plt.title(f"{col} (Before Capping)")
plt.tight_layout()
plt.show()
 """
# Visualize after capping
""" plt.figure(figsize=(15, 10))
for i, col in enumerate(skewed_cols):
    plt.subplot(2, 2, i + 1)
    sb.boxplot(x=df[col])
    plt.title(f"{col} (After Capping)")
plt.tight_layout()
plt.show() """

# Calculate and print new skewness after capping outliers
for col in skewed_cols:
    skewness = stats.skew(df[col])
    print(f"Skewness of {col}: {skewness}")

# List your columns
cols_to_transform = ['Economic_Impact_Million_Usd', 'Machinery_Per_Ag_Land', 'Ag_Land_Index', 'Labor_Index']

# Apply Yeo-Johnson Transformation
pt = PowerTransformer(method='yeo-johnson')
df[cols_to_transform] = pt.fit_transform(df[cols_to_transform])

for col in cols_to_transform:
    print(f"New Skewness of {col}: {skew(df[col]):.3f}")

numeric_cols_transformed = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# Calculate and print skewness
for col in numeric_cols_transformed:
    skewness = stats.skew(df[col])
    print(f"Skewness of {col}: {skewness}")

print(numeric_cols_transformed)

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply scaling to the numerical columns
df[numeric_cols_transformed] = scaler.fit_transform(df[numeric_cols_transformed])
#print(df.head())

# Calculate the correlation matrix
correlation_matrix = df.corr(numeric_only=True)  # Use numeric_only to avoid warning

""" # Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Create the scatter plot
plt.figure(figsize=(10, 6))
sb.scatterplot(x='Economic_Impact_Million_Usd', y='Crop_Yield_Mt_Per_Ha', data=df)
plt.title('Scatter Plot: Economic Impact vs. Crop Yield')
plt.xlabel('Economic Impact (Million USD)')
plt.ylabel('Crop Yield (MT per HA)')
plt.grid(True)
plt.show() """

from sklearn.preprocessing import LabelEncoder

y = df['Crop_Yield_Mt_Per_Ha']  
X = df.drop(columns=['Crop_Yield_Mt_Per_Ha'])  # Keep only independent variables
feature_names = X.columns.tolist()     

# Use separate label encoders for each categorical column
le_country = LabelEncoder()
le_region = LabelEncoder()
le_crop = LabelEncoder()
le_strategy = LabelEncoder()
le_year = LabelEncoder()

X['Country'] = le_country.fit_transform(X['Country'])
X['Region'] = le_region.fit_transform(X['Region'])
X['Crop_Type'] = le_crop.fit_transform(X['Crop_Type'])
X['Adaptation_Strategies'] = le_strategy.fit_transform(X['Adaptation_Strategies'])
X['Year'] = le_year.fit_transform(X['Year'])

#Train the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import GradientBoostingRegressor


# Initialize the model
gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Fit the model
gbm_model.fit(X_train, y_train)

# Make predictions
y_gbm_pred = gbm_model.predict(X_test)

# Evaluate the model
gbm_r2 = r2_score(y_test, y_gbm_pred)
gbm_mae = mean_absolute_error(y_test, y_gbm_pred)
gbm_mse = mean_squared_error(y_test, y_gbm_pred)

# Print the results
print(f"R² Score: {gbm_r2:.4f}")
print(f"Mean Absolute Error: {gbm_mae:.4f}")
print(f"Mean Squared Error: {gbm_mse:.4f}")

# Make predictions for the entire dataset
y_gbm_pred_full = gbm_model.predict(X)

# Add predictions to the full dataset
decoded_X_full = X.copy()
decoded_X_full['Predicted_Crop_Yield'] = y_gbm_pred_full
decoded_X_full['Country'] = le_country.inverse_transform(X['Country'])
decoded_X_full['Region'] = le_region.inverse_transform(X['Region'])
decoded_X_full['Crop_Type'] = le_crop.inverse_transform(X['Crop_Type'])
decoded_X_full['Adaptation_Strategies'] = le_strategy.inverse_transform(X['Adaptation_Strategies'])
decoded_X_full['Year'] = le_year.inverse_transform(X['Year'])
decoded_X_full['Next_Year'] = decoded_X_full['Year'].astype(int) + 1

# Export the full dataset with predictions
decoded_X_full.to_csv('decoded_X_full.csv', index=False)

# Display or export
print(decoded_X_full.head())

