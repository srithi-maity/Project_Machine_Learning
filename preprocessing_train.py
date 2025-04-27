import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# Load the final cleaned dataset
df = pd.read_json('train_cleaned.json', lines=True)

# 1. Basic Info
print("Basic Info:")
print(df.shape)
print(df.columns)
print(df.dtypes)

# 2. Missing Values
print("Missing Values:")
print(df.isnull().sum())

# Interpretation:
# --> If missing values exist, decide to fill or drop. Otherwise, proceed safely.

# 3. Histograms of all numerical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

df[numeric_cols].hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Interpretation:
# --> Helps to see skewed features, distributions, and identify normal or extreme behaviors.

# 4. Correlation Heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()

# Interpretation:
# --> High correlation (bright or dark regions) hints at redundant features.

# 5. Outlier Detection using Z-score
z_scores = stats.zscore(df[numeric_cols])
outliers = (np.abs(z_scores) > 3)
print("Number of Outliers per Feature:")
print(outliers.sum(axis=0))

# Interpretation:
# --> Features with many outliers may need winsorization, clipping, or removal.

# 6. Sequence Length Analysis
plt.figure(figsize=(8,6))
sns.histplot(df['seq_length'], kde=True)
plt.title('Distribution of RNA Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.show()

# Interpretation:
# --> Helps see if most RNAs are short/long, or if data has a fixed design.

# 7. Structure and Loop Type Analysis
plt.figure(figsize=(8,6))
sns.countplot(x=df['predicted_loop_type'].str[0])  # Look at first loop type character
plt.title('Primary Loop Type Distribution')
plt.xlabel('Loop Type')
plt.ylabel('Count')
plt.show()

# Interpretation:
# --> Tells dominant secondary structures (Hairpins, Stems, etc.)

# 8. Signal-to-Noise Ratio
plt.figure(figsize=(8,6))
sns.histplot(df['signal_to_noise'], kde=True)
plt.title('Signal-to-Noise Ratio Distribution')
plt.xlabel('Signal-to-Noise')
plt.ylabel('Count')
plt.show()

# Interpretation:
# --> High SNR = better quality sequences. Low SNR = possibly noisy data.

# 9. Boxplots for Reactivity and Degradation
reactivity_cols = [col for col in df.columns if 'reactivity' in col or 'deg_' in col]

plt.figure(figsize=(20,8))
sns.boxplot(data=df[reactivity_cols], orient='h')
plt.title('Boxplots: Reactivity and Degradation Measures')
plt.show()

# Interpretation:
# --> Boxplots show variability and presence of extreme values in reactivity/degradation experiments.

# 10. Scaling Numeric Features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])
df_scaled = pd.DataFrame(scaled_data, columns=numeric_cols)
print("Data Scaling Completed. Ready for ML Models!")

# Define Features and Target
X = df_scaled
y = df['target']  # Assuming 'target' is your target column

# 1. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. XGBoost Model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)

# 3. Hyperparameter Tuning using GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 10],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best Hyperparameters found: ", grid_search.best_params_)

# 4. Train the Model with Best Parameters
best_model = grid_search.best_estimator_

# 5. Make Predictions
y_pred = best_model.predict(X_test)

# 6. Evaluate the Model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))

# 7. Feature Importance
xgb.plot_importance(best_model)
plt.title('XGBoost Feature Importance')
plt.show()