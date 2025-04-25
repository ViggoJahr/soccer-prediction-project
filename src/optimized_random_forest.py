import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
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

# Define Random Forest with RandomizedSearchCV
param_dist_rf = {
    'n_estimators': [150, 200, 250],
    'max_depth': [20, 30, 40],
    'min_samples_split': [3, 5, 7],
    'min_samples_leaf': [1, 2],
    'max_features': ['log2', 'sqrt']
}
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf, n_iter=15, cv=2, 
                                     scoring='accuracy', n_jobs=2, random_state=42, verbose=1)
random_search_rf.fit(X_train_scaled, y_train)

# Best Random Forest
best_rf = random_search_rf.best_estimator_
print(f"\nRandom Forest Best parameters: {random_search_rf.best_params_}")
print(f"Random Forest Best cross-validation accuracy: {random_search_rf.best_score_:.4f}")

# Evaluate Random Forest
try:
    y_pred = best_rf.predict(X_test_scaled)
    y_pred_proba = best_rf.predict_proba(X_test_scaled)
    # Normalize probabilities
    y_pred_proba_normalized = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    # Print metrics
    print("\nRandom Forest Results:")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Log loss: {log_loss(y_test, y_pred_proba_normalized):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))
    
    # Feature importance
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nRandom Forest Feature importance:")
    print(importances)
    
    # Visualizations
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Away Win', 'Draw', 'Home Win'], 
                yticklabels=['Away Win', 'Draw', 'Home Win'])
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(output_dir / 'random_forest_confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to visualizations/random_forest_confusion_matrix.png")

    # 2. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(['Away Win', 'Draw', 'Home Win']):
        precision, recall, _ = precision_recall_curve(y_test == i, y_pred_proba_normalized[:, i])
        plt.plot(recall, precision, label=f'{label}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Random Forest Precision-Recall Curves')
    plt.legend()
    plt.savefig(output_dir / 'random_forest_precision_recall.png')
    plt.close()
    print("Precision-recall curves saved to visualizations/random_forest_precision_recall.png")

    # 3. Calibration Plot
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(['Away Win', 'Draw', 'Home Win']):
        prob_true, prob_pred = calibration_curve(y_test == i, y_pred_proba_normalized[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f'{label}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Random Forest Calibration Plot')
    plt.legend()
    plt.savefig(output_dir / 'random_forest_calibration.png')
    plt.close()
    print("Calibration plot saved to visualizations/random_forest_calibration.png")

    # 4. Feature Importance Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importances.head(15), palette='viridis')
    plt.title('Random Forest Top 15 Feature Importance')
    plt.savefig(output_dir / 'random_forest_feature_importance.png')
    plt.close()
    print("Feature importance plot saved to visualizations/random_forest_feature_importance.png")

    # Analyze errors
    errors = df.iloc[y_test.index][y_test != y_pred].copy()
    errors['Predicted'] = y_pred[y_test != y_pred]
    errors['Correct'] = y_test[y_test != y_pred] == y_pred[y_test != y_pred]
    full_test_df = df.iloc[y_test.index].copy()
    full_test_df['Correct'] = y_test == y_pred
    full_test_df['Max_Prob'] = y_pred_proba_normalized.max(axis=1)
    print("\nSample misclassified matches (Random Forest):")
    print(errors[['home_team_api_id', 'away_team_api_id', 'result', 'home_prob', 'draw_prob', 'away_prob', 'Predicted']].head())
    
    # 5. Prediction Probability Distribution
    plt.figure(figsize=(10, 6))
    for prob in ['home_prob', 'draw_prob', 'away_prob']:
        sns.kdeplot(full_test_df[full_test_df['Correct']][prob], label=f'{prob} (Correct)', fill=False)
        sns.kdeplot(errors[prob], label=f'{prob} (Incorrect)', fill=True, alpha=0.3)
    plt.title('Probability Distribution: Correct vs. Incorrect Predictions')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(output_dir / 'random_forest_pred_prob_distribution.png')
    plt.close()
    print("Prediction probability distribution saved to visualizations/random_forest_pred_prob_distribution.png")

    # 6. Prediction Confidence vs. Accuracy
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Max_Prob', y='Correct', data=full_test_df, alpha=0.5)
    plt.title('Prediction Confidence vs. Accuracy')
    plt.xlabel('Maximum Predicted Probability')
    plt.ylabel('Correct Prediction (1=Correct, 0=Incorrect)')
    plt.savefig(output_dir / 'random_forest_confidence_accuracy.png')
    plt.close()
    print("Prediction confidence vs. accuracy saved to visualizations/random_forest_confidence_accuracy.png")

    # 7. Misclassified Matches by Elo Difference
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Correct', y='elo_diff', data=full_test_df)
    plt.title('Elo Difference for Correct vs. Incorrect Predictions')
    plt.xlabel('Prediction Correctness')
    plt.ylabel('Elo Difference (Home - Away)')
    plt.xticks([0, 1], ['Incorrect', 'Correct'])
    plt.savefig(output_dir / 'random_forest_elo_diff_boxplot.png')
    plt.close()
    print("Elo difference boxplot saved to visualizations/random_forest_elo_diff_boxplot.png")

    # 8. Misclassified Probability Stacked Bar (for actual Draws)
    draw_errors = errors[errors['result'] == 1]  # Actual Draws
    draw_pred_counts = draw_errors['Predicted'].value_counts(normalize=True)
    draw_pred_df = pd.DataFrame({'Class': ['Away Win', 'Draw', 'Home Win'], 'Proportion': [0, 0, 0]})
    for idx, count in draw_pred_counts.items():
        draw_pred_df.loc[idx, 'Proportion'] = count
    plt.figure(figsize=(8, 6))
    draw_pred_df.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Predicted Classes for Actual Draw Outcomes')
    plt.xlabel('Predicted Class')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)
    plt.savefig(output_dir / 'random_forest_draw_misclassification.png')
    plt.close()
    print("Draw misclassification bar plot saved to visualizations/random_forest_draw_misclassification.png")

    # 9. Temporal Error Analysis
    full_test_df['date'] = df.iloc[y_test.index]['date']
    full_test_df['Year'] = full_test_df['date'].dt.year
    error_rate_by_year = full_test_df.groupby('Year')['Correct'].mean()
    plt.figure(figsize=(10, 6))
    error_rate_by_year.plot(kind='line', marker='o')
    plt.title('Prediction Accuracy by Year')
    plt.xlabel('Year')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(output_dir / 'random_forest_temporal_accuracy.png')
    plt.close()
    print("Temporal accuracy plot saved to visualizations/random_forest_temporal_accuracy.png")

except Exception as e:
    print(f"Error evaluating Random Forest: {e}")

# Save predictions
try:
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    dataprediction_path = Path(__file__).resolve().parents[1] / "data" / "predictions.csv"
    results.to_csv(dataprediction_path, index=False)
    print("\nPredictions saved to predictions.csv")
except Exception as e:
    print(f"Error saving predictions: {e}")