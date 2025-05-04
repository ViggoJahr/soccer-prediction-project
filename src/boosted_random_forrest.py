import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix, precision_recall_curve
from sklearn.calibration import calibration_curve
from pathlib import Path

# Set up output directory for visualizations
output_dir = Path(__file__).resolve().parents[1] / "data" / "visualizations"
output_dir.mkdir(exist_ok=True)

# Load dataset from relative path
dataset_path = Path(__file__).resolve().parents[1] / "data" / "football_ml_dataset.csv"
try:
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} matches")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Add draw rate features
df['team_draw_rate'] = df.groupby('home_team_api_id')['result'].transform(lambda x: (x == 0).mean())
df['recent_draw_rate'] = df.groupby('home_team_api_id')['result'].transform(
    lambda x: (x == 0).rolling(5, min_periods=1).mean().shift(1)).fillna(0)
df['elo_diff_squared'] = df['elo_diff'] ** 2

# Ensure date column is parsed for temporal analysis
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Features (dropped low-importance tactical features)
features = [
    'home_prob', 'draw_prob', 'away_prob', 'home_prob_median', 'draw_prob_median', 'away_prob_median',
    'home_elo', 'away_elo', 'elo_diff', 'elo_diff_squared',
    'home_form_points', 'away_form_points', 'home_form_goals', 'away_form_goals',
    'home_form_intensity', 'away_form_intensity', 'home_form_draw', 'away_form_draw',
    'h2h_home_wins', 'h2h_goal_diff', 'h2h_home_win_rate', 'h2h_home_goals', 'h2h_away_goals',
    'h2h_home_goal_efficiency', 'h2h_draw_rate',
    'league_id', 'team_draw_rate', 'recent_draw_rate'
]
X = df[features]
y = df['result'].map({-1: 0, 0: 1, 1: 2})  # Map to 0 (Away), 1 (Draw), 2 (Home)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define XGBoost with RandomizedSearchCV
param_dist_xgb = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}
# Use GPU if available (optional, remove if no compatible GPU/xgboost build)
# gpu_params = {'tree_method': 'hist', 'device': 'cuda'} if XGBClassifier()._get_param_meta()['device'] == 'cuda' else {}
# xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', **gpu_params) # Removed RF
xgb = XGBClassifier(random_state=42, eval_metric='mlogloss') # Removed RF

# Use Stratified K-Fold for more robust CV
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search_xgb = RandomizedSearchCV(xgb, param_distributions=param_dist_xgb, n_iter=25, cv=cv_strategy, # Increased n_iter and changed cv
                                      scoring='accuracy', n_jobs=-1, random_state=42, verbose=0) # Use all cores

# Define evaluation set for early stopping
eval_set = [(X_test_scaled, y_test)]

# Fit with early stopping
random_search_xgb.fit(X_train_scaled, y_train,
                      verbose=False) # Suppress XGBoost's own verbose output during search

# Best XGBoost
best_xgb = random_search_xgb.best_estimator_
print(f"XGBoost Best parameters: {random_search_xgb.best_params_}")
print(f"XGBoost Best cross-validation accuracy: {random_search_xgb.best_score_:.4f}")

# Evaluate XGBoost
try:
    y_pred = best_xgb.predict(X_test_scaled)
    y_pred_proba = best_xgb.predict_proba(X_test_scaled)
    # Normalize probabilities (though XGBoost proba should sum to 1)
    y_pred_proba_normalized = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

    # Print metrics
    print("\nXGBoost Results:")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Log loss: {log_loss(y_test, y_pred_proba_normalized):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))

    # Feature importance
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': best_xgb.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nXGBoost Feature importance:")
    print(importances)

    # Visualizations
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Away Win', 'Draw', 'Home Win'],
                yticklabels=['Away Win', 'Draw', 'Home Win'])
    plt.title('XGBoost Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(output_dir / 'xgboost_confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to visualizations/xgboost_confusion_matrix.png")

    # 2. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(['Away Win', 'Draw', 'Home Win']):
        precision, recall, _ = precision_recall_curve(y_test == i, y_pred_proba_normalized[:, i])
        plt.plot(recall, precision, label=f'{label}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('XGBoost Precision-Recall Curves')
    plt.legend()
    plt.savefig(output_dir / 'xgboost_precision_recall.png')
    plt.close()
    print("Precision-recall curves saved to visualizations/xgboost_precision_recall.png")

    # 3. Calibration Plot
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(['Away Win', 'Draw', 'Home Win']):
        prob_true, prob_pred = calibration_curve(y_test == i, y_pred_proba_normalized[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f'{label}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('XGBoost Calibration Plot')
    plt.legend()
    plt.savefig(output_dir / 'xgboost_calibration.png')
    plt.close()
    print("Calibration plot saved to visualizations/xgboost_calibration.png")

    # 4. Feature Importance Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importances.head(15), palette='viridis')
    plt.title('XGBoost Top 15 Feature Importance')
    plt.savefig(output_dir / 'xgboost_feature_importance.png')
    plt.close()
    print("Feature importance plot saved to visualizations/xgboost_feature_importance.png")

    # Analyze errors
    errors = df.iloc[y_test.index][y_test != y_pred].copy()
    errors['Predicted'] = y_pred[y_test != y_pred]
    errors['Correct'] = y_test[y_test != y_pred] == y_pred[y_test != y_pred]
    errors['Actual'] = y_test[y_test != y_pred]
    full_test_df = df.iloc[y_test.index].copy()
    full_test_df['Correct'] = y_test == y_pred
    full_test_df['Max_Prob'] = y_pred_proba_normalized.max(axis=1)
    full_test_df['Predicted'] = y_pred
    full_test_df['Actual'] = y_test

    print("\nSample misclassified matches (XGBoost):")
    print(errors[['home_team_api_id', 'away_team_api_id', 'result', 'Actual', 'Predicted', 'home_prob', 'draw_prob', 'away_prob']].head())

    # 5. Prediction Probability Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(full_test_df[full_test_df['Correct']]['Max_Prob'], label='Correct', fill=True, alpha=0.3)
    sns.kdeplot(full_test_df[~full_test_df['Correct']]['Max_Prob'], label='Incorrect', fill=True, alpha=0.3)
    plt.title('Max Prediction Probability Distribution: Correct vs. Incorrect (XGBoost)')
    plt.xlabel('Maximum Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(output_dir / 'xgboost_pred_prob_distribution.png')
    plt.close()
    print("Prediction probability distribution saved to visualizations/xgboost_pred_prob_distribution.png")

    # 6. Prediction Confidence vs. Accuracy
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Max_Prob', y='Correct', data=full_test_df, alpha=0.5, hue='Correct', legend=False)
    try:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess
        ys = lowess(full_test_df['Correct'], full_test_df['Max_Prob'], frac=0.3)
        plt.plot(ys[:, 0], ys[:, 1], color='red', lw=2, label='Smoothed Accuracy Trend')
        plt.legend()
    except ImportError:
        print("Install statsmodels for smoothed accuracy trend line.")
    plt.title('Prediction Confidence vs. Accuracy (XGBoost)')
    plt.xlabel('Maximum Predicted Probability')
    plt.ylabel('Correct Prediction (1=Correct, 0=Incorrect)')
    plt.ylim(-0.1, 1.1)
    plt.savefig(output_dir / 'xgboost_confidence_accuracy.png')
    plt.close()
    print("Prediction confidence vs. accuracy saved to visualizations/xgboost_confidence_accuracy.png")

    # 7. Misclassified Matches by Elo Difference
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Correct', y='elo_diff', data=full_test_df)
    plt.title('Elo Difference for Correct vs. Incorrect Predictions (XGBoost)')
    plt.xlabel('Prediction Correctness')
    plt.ylabel('Elo Difference (Home - Away)')
    plt.xticks([0, 1], ['Incorrect', 'Correct'])
    plt.savefig(output_dir / 'xgboost_elo_diff_boxplot.png')
    plt.close()
    print("Elo difference boxplot saved to visualizations/xgboost_elo_diff_boxplot.png")

    # 8. Misclassified Probability Stacked Bar (Improved logic)
    misclassification_summary = full_test_df[~full_test_df['Correct']].groupby('Actual')['Predicted'].value_counts(normalize=True).unstack(fill_value=0)
    misclassification_summary = misclassification_summary.rename(columns={0: 'Away Win', 1: 'Draw', 2: 'Home Win'},
                                                                 index={0: 'Actual Away Win', 1: 'Actual Draw', 2: 'Actual Home Win'})

    plt.figure(figsize=(10, 7))
    misclassification_summary.plot(kind='bar', stacked=True, figsize=(10, 7), colormap='viridis')
    plt.title('Distribution of Incorrect Predictions by Actual Outcome (XGBoost)')
    plt.xlabel('Actual Outcome')
    plt.ylabel('Proportion of Incorrect Predictions')
    plt.xticks(rotation=0)
    plt.legend(title='Predicted Outcome')
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_misclassification_by_actual.png')
    plt.close()
    print("Misclassification analysis plot saved to visualizations/xgboost_misclassification_by_actual.png")

    # 9. Temporal Error Analysis
    full_test_df['date'] = df.iloc[y_test.index]['date']
    full_test_df['Year'] = full_test_df['date'].dt.year
    accuracy_by_year = full_test_df.groupby('Year')['Correct'].mean()
    plt.figure(figsize=(10, 6))
    accuracy_by_year.plot(kind='line', marker='o')
    plt.title('Prediction Accuracy by Year (XGBoost)')
    plt.xlabel('Year')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(bottom=accuracy_by_year.min() - 0.05, top=accuracy_by_year.max() + 0.05)
    plt.savefig(output_dir / 'xgboost_temporal_accuracy.png')
    plt.close()
    print("Temporal accuracy plot saved to visualizations/xgboost_temporal_accuracy.png")

except Exception as e:
    print(f"Error evaluating XGBoost: {e}")

# Save predictions
try:
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Prob_Away': y_pred_proba_normalized[:, 0],
        'Prob_Draw': y_pred_proba_normalized[:, 1],
        'Prob_Home': y_pred_proba_normalized[:, 2]
    })

    dataprediction_path = Path(__file__).resolve().parents[1] / "data" / "xgboost_predictions.csv"
    results.to_csv(dataprediction_path, index=False)
    print(f"\nPredictions saved to {dataprediction_path.name}")
except Exception as e:
    print(f"Error saving predictions: {e}")