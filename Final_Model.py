import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ====================== 1. RAW DATA ======================
print("\n=== STEP 1: Raw Data Loading ===")
train = pd.read_json('train.json', lines=True)
test = pd.read_json('test.json', lines=True)
print(f"Loaded {train.shape} training and {test.shape} test samples")
print(f"Train data contains these columns {train.columns}")
print(f"Test data contains these columns {test.columns}")


# ====================== 2. EDA ======================
print("\n=== STEP 2: Exploratory Data Analysis ===")
print("\n--- Sequence Length Analysis ---")
print(train['seq_length'].describe())

print("\n--- Reactivity Analysis ---")
reactivity_stats = train['reactivity'].apply(lambda x: pd.Series({
    'mean': np.mean(x),
    'min': np.min(x),
    'max': np.max(x),
    'std': np.std(x)
}))
print(reactivity_stats.describe())

###___________Plots___________###
print(f"======Plotting Reactivity Distribution======")

plt.figure(figsize=(8, 4))
train['reactivity'].apply(np.mean).hist(bins=50, color='skyblue')
plt.title('Distribution of Mean Reactivity')
plt.xlabel('Reactivity')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()  ##  Insight: Most reactivity values cluster between 0.2-0.6 ,Long tail indicates potential outliers

print(f"=======Plotting Sequence Length Analysis=======")

plt.figure(figsize=(8, 4))
train['seq_length'].hist(bins=30, color='lightgreen')
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Length')
plt.ylabel('Count')
plt.grid(False)
plt.show() ##Insight: Sequences vary in length (max=130)

print(f"========Plotting Reactivity by Loop Type========")

loop_types = train['predicted_loop_type'].apply(lambda x: x[0])  # First character
mean_reactivity = train['reactivity'].apply(np.mean)
pd.DataFrame({'LoopType': loop_types, 'Reactivity': mean_reactivity}) \
  .groupby('LoopType') \
  .mean() \
  .plot(kind='bar', color='salmon')
plt.title('Mean Reactivity by Loop Type')
plt.ylabel('Reactivity')
plt.show() ##Insight: Loop types 'E' (external) show higher reactivity ,Helps prioritize important loop types

# ====================== 3. PREPROCESSING ======================
print("\n=== STEP 3: Preprocessing ===")
print("Validating BPP files...")
bpps_dir = 'bpps'
train_ids = set(train['id'])
test_ids = set(test['id'])
bpps_files = {f.split('.')[0] for f in os.listdir(bpps_dir) if f.endswith('.npy')}

assert train_ids <= bpps_files, "Missing BPP files for train IDs"
assert test_ids <= bpps_files, "Missing BPP files for test IDs"
print("All BPP files validated")

# Calculate max sequence length
max_seq_len = max(train['sequence'].apply(len).max(), test['sequence'].apply(len).max())
print(f"\nMax sequence length: {max_seq_len}")

# ====================== 4. FEATURE ENGINEERING ======================
print("\n=== STEP 4: Feature Engineering ===")


def pad_sequence(seq, max_len, pad_value=0):
    return seq + [pad_value] * (max_len - len(seq))


# --------------------- 4.1 Sequence/Structure Encoding ---------------------
print("\n--- Sequence/Structure Encoding ---")


def encode_features(df, max_len):
    # Base encoding (0-3)
    seq_encoded = np.stack(df['sequence'].apply(
        lambda x: pad_sequence([{'A': 0, 'C': 1, 'G': 2, 'U': 3}[c] for c in x],
                               max_len, pad_value=-1)))

    # Structure encoding with offset 10 (10-12)
    struct_encoded = np.stack(df['structure'].apply(
        lambda x: pad_sequence([{'.': 10, '(': 11, ')': 12}[c] for c in x],
                               max_len, pad_value=-1)))

    # Loop encoding with offset 20 (20-26)
    loop_map = {'S': 20, 'M': 21, 'I': 22, 'B': 23, 'H': 24, 'E': 25, 'X': 26}
    loop_encoded = np.stack(df['predicted_loop_type'].apply(
        lambda x: pad_sequence([loop_map[c] for c in x],
                               max_len, pad_value=-1)))

    return np.concatenate([seq_encoded, struct_encoded, loop_encoded], axis=1)


X_train_encoded = encode_features(train, max_seq_len)
X_test_encoded = encode_features(test, max_seq_len)

# --------------------- 4.2 BPP Features ---------------------
print("\n--- BPP Feature Extraction ---")


def get_bpp_features(id_):
    bpp = np.load(f'bpps/{id_}.npy')
    upper_tri = bpp[np.triu_indices_from(bpp, k=1)]
    return {
        'bpp_max': np.max(upper_tri),
        'bpp_mean': np.mean(upper_tri),
        'bpp_std': np.std(upper_tri),
        'bpp_sum': np.sum(upper_tri),
        'bpp_entropy': -np.sum(upper_tri * np.log(upper_tri + 1e-10))
    }


train_bpp = pd.DataFrame(train['id'].apply(get_bpp_features).tolist())
test_bpp = pd.DataFrame(test['id'].apply(get_bpp_features).tolist())




print(f"=======Plotting BPP Feature Correlations=======")

import seaborn as sns
bpp_corr = train_bpp.corr()
plt.figure(figsize=(6, 4))
sns.heatmap(bpp_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between BPP Features')
plt.show() ##Insight: bpp_max and bpp_sum are highly correlated (may need feature selection)



# --------------------- 4.3 Feature Combination ---------------------
print("\n--- Feature Combination ---")
X_train = np.concatenate([X_train_encoded, train[['seq_length']].values, train_bpp.values], axis=1)
X_test = np.concatenate([X_test_encoded, test[['seq_length']].values, test_bpp.values], axis=1)
print(f"Final feature shapes - Train: {X_train.shape}, Test: {X_test.shape}")

# ====================== 5. MODEL TRAINING ======================
print("\n=== STEP 5: XGBoost Training ===")
y_train = train['reactivity'].apply(np.mean).values
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=20,
    eval_metric='rmse'
)

model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=10)

# ====================== 6. EVALUATION ======================
print("\n=== STEP 6: Evaluation ===")
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# Feature Importance
plot_importance(model, max_num_features=20)
plt.title('Feature Importance')
plt.show()

# Test Predictions
y_test_pred = model.predict(X_test)
print("\nTest predictions sample:", y_test_pred[:5])






print(f"*******************************************************************************")

print(f"============ Model Improvement through Hyperparameter Tuning ============== ")

print(f"_______Step 1: Prepare for Hyperparameter Tuning_______")

from sklearn.model_selection import KFold

# Prepare full training data
X = X_train
y = y_train

# Set up KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


print(f"_________Step 2: Define Parameter Search Space _________")
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'min_child_weight': [1, 2, 3, 4],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [0, 0.1, 1, 10]
}

xgb = XGBRegressor(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=50,
    scoring='neg_root_mean_squared_error',
    cv=kf,
    verbose=2,
    random_state=42
)

random_search.fit(X, y)

print("Best parameters:", random_search.best_params_)
print("Best RMSE:", -random_search.best_score_)

print(f"===============Train Final Model with Best Parameters=================")
final_model = XGBRegressor(
    **random_search.best_params_,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20
)

final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=10)

# Make predictions on validation data
y_val_pred = final_model.predict(X_val)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Compute metrics
final_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
final_mae = mean_absolute_error(y_val, y_val_pred)
final_r2 = r2_score(y_val, y_val_pred)

# Print them
print("\n=== Final Model Evaluation on Validation Set ===")
print(f"Final RMSE: {final_rmse:.4f}")
print(f"Final MAE: {final_mae:.4f}")
print(f"Final R² Score: {final_r2:.4f}")


# Predict on test data
final_test_preds = final_model.predict(X_test)

# Show sample predictions
print("\nSample Test Predictions from Final Model:")
print(final_test_preds[:5])


from xgboost import plot_importance
import matplotlib.pyplot as plt

plot_importance(final_model, max_num_features=20)
plt.title('Final Model Feature Importance')
plt.tight_layout()
plt.show()
