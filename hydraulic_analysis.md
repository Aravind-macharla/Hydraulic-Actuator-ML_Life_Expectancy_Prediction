```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Set style
sns.set(color_codes=True)
plt.rcParams['figure.figsize'] = (15, 10)

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================

print("="*70)
print("HYDRAULIC SYSTEM CONDITION MONITORING - DECISION TREE APPROACH")
print("="*70)

# Download and extract data (if needed)
# !wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00447/data.zip"
# !unzip -o "data.zip"

# Import sensor values as features
sensor_names = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
                "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3",
                "TS4", "VS1", "CE", "CP", "SE"]

feature_list = []
for sensor in sensor_names:
    data = np.genfromtxt(f"{sensor}.txt")
    feature_list.append(data)

# Extract mean values
feature_means = {}
for i in range(len(sensor_names)):
    feature_means[sensor_names[i]] = feature_list[i].mean(axis=1)

# Create features dataframe
df_features = pd.DataFrame(feature_means)

# Import target labels
target = np.genfromtxt(r"profile.txt")
df_targets = pd.DataFrame(target, columns=[
    "Cooler_Condition",
    "Valve_Condition",
    "Internal_Pump_Leakage",
    "Hydraulic_Accumulator",
    "Stable_Flag"
])

# Combine features and targets
df_final = pd.concat([df_features, df_targets], axis=1)

print(f"\nDataset shape: {df_final.shape}")
print(f"Number of samples: {len(df_final)}")
print(f"Number of features: {len(sensor_names)}")

# Prepare features and targets
X = df_final.iloc[:, :-5]
targets = {
    'Cooler_Condition': df_final['Cooler_Condition'].astype(int),
    'Valve_Condition': df_final['Valve_Condition'].astype(int),
    'Internal_Pump_Leakage': df_final['Internal_Pump_Leakage'].astype(int),
    'Hydraulic_Accumulator': df_final['Hydraulic_Accumulator'].astype(int),
    'Stable_Flag': df_final['Stable_Flag'].astype(int)
}

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, target_name):
    """Comprehensive model evaluation"""

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_precision = precision_score(y_train, y_pred_train, average='weighted')
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    train_recall = recall_score(y_train, y_pred_train, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    train_f1 = f1_score(y_train, y_pred_train, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')

    print(f"\n{'='*70}")
    print(f"RESULTS FOR: {target_name}")
    print(f"{'='*70}")
    print(f"\nTraining Set Metrics:")
    print(f"  Accuracy:  {train_accuracy:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")

    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")

    # Check for overfitting
    overfit_gap = train_accuracy - test_accuracy
    if overfit_gap > 0.1:
        print(f"\n⚠️  Warning: Possible overfitting detected (gap: {overfit_gap:.4f})")
    else:
        print(f"\n✓ Good generalization (gap: {overfit_gap:.4f})")

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'predictions': y_pred_test
    }

def plot_confusion_matrix(y_true, y_pred, target_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {target_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{target_name.replace(" ", "_")}.png', dpi=300)
    print(f"\n✓ Confusion matrix saved as 'confusion_matrix_{target_name.replace(' ', '_')}.png'")

def plot_decision_tree(model, feature_names, class_names, target_name, max_depth=3):
    """Visualize decision tree"""
    plt.figure(figsize=(20, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=max_depth  # Limit depth for readability
    )
    plt.title(f'Decision Tree Visualization - {target_name}\n(Showing top {max_depth} levels)',
              fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'decision_tree_{target_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Decision tree saved as 'decision_tree_{target_name.replace(' ', '_')}.png'")

def plot_feature_importance(model, feature_names, target_name, top_n=10):
    """Plot feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances - {target_name}')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{target_name.replace(" ", "_")}.png', dpi=300)
    print(f"✓ Feature importance saved as 'feature_importance_{target_name.replace(' ', '_')}.png'")

    print(f"\nTop {top_n} Most Important Features:")
    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")

# ============================================================================
# 3. TRAIN DECISION TREE MODELS FOR ALL TARGETS
# ============================================================================

results_summary = {}

for target_name, y in targets.items():

    print(f"\n\n{'#'*70}")
    print(f"# TRAINING MODEL FOR: {target_name}")
    print(f"{'#'*70}")

    # Check class distribution
    print(f"\nClass distribution:")
    print(y.value_counts().sort_index())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )

    # ========================================================================
    # DECISION TREE MODEL
    # ========================================================================

    # Create decision tree classifier
    dt_model = DecisionTreeClassifier(
        max_depth=10,              # Limit depth to prevent overfitting
        min_samples_split=20,      # Minimum samples required to split
        min_samples_leaf=10,       # Minimum samples required at leaf
        max_features='sqrt',       # Number of features to consider
        random_state=1,
        criterion='gini'           # Use Gini impurity
    )

    # Train the model
    print(f"\n{'─'*70}")
    print("Training Decision Tree Model...")
    print(f"{'─'*70}")
    dt_model.fit(X_train, y_train)
    print("✓ Training completed!")

    # Evaluate model
    results = evaluate_model(dt_model, X_train, X_test, y_train, y_test, target_name)
    results_summary[target_name] = results

    # Cross-validation
    print(f"\n{'─'*70}")
    print("Cross-Validation (5-fold):")
    print(f"{'─'*70}")
    cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  CV Accuracy Scores: {cv_scores}")
    print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Visualizations
    print(f"\n{'─'*70}")
    print("Generating Visualizations...")
    print(f"{'─'*70}")

    # 1. Confusion Matrix
    plot_confusion_matrix(y_test, results['predictions'], target_name)

    # 2. Feature Importance
    plot_feature_importance(dt_model, X.columns.tolist(), target_name, top_n=10)

    # 3. Decision Tree Visualization (only for first 3 levels)
    class_names = sorted(y.unique())
    plot_decision_tree(dt_model, X.columns.tolist(), class_names, target_name, max_depth=3)

    # Classification Report
    print(f"\n{'─'*70}")
    print("Detailed Classification Report:")
    print(f"{'─'*70}")
    print(classification_report(y_test, results['predictions']))

    # Model complexity metrics
    print(f"\n{'─'*70}")
    print("Model Complexity:")
    print(f"{'─'*70}")
    print(f"  Tree Depth: {dt_model.get_depth()}")
    print(f"  Number of Leaves: {dt_model.get_n_leaves()}")
    print(f"  Number of Features Used: {np.sum(dt_model.feature_importances_ > 0)}")

# ============================================================================
# 4. SUMMARY OF ALL MODELS
# ============================================================================

print(f"\n\n{'='*70}")
print("OVERALL SUMMARY - ALL TARGETS")
print(f"{'='*70}\n")

summary_df = pd.DataFrame({
    target: {
        'Train Accuracy': f"{results['train_accuracy']:.4f}",
        'Test Accuracy': f"{results['test_accuracy']:.4f}",
        'Train Precision': f"{results['train_precision']:.4f}",
        'Test Precision': f"{results['test_precision']:.4f}"
    }
    for target, results in results_summary.items()
}).T

print(summary_df)

# Best and worst performing models
print(f"\n{'─'*70}")
test_accuracies = {k: v['test_accuracy'] for k, v in results_summary.items()}
best_model = max(test_accuracies, key=test_accuracies.get)
worst_model = min(test_accuracies, key=test_accuracies.get)

print(f"Best Performing Model: {best_model} ({test_accuracies[best_model]:.4f})")
print(f"Worst Performing Model: {worst_model} ({test_accuracies[worst_model]:.4f})")

# ============================================================================
# 5. SAVE FINAL MODELS (Optional)
# ============================================================================

print(f"\n{'─'*70}")
print("Saving Models...")
print(f"{'─'*70}")

import pickle

for target_name, y in targets.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )

    # Train final model on all training data
    final_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=1
    )
    final_model.fit(X_train, y_train)

    # Save model
    filename = f'dt_model_{target_name.replace(" ", "_")}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"✓ Model saved: {filename}")

print(f"\n{'='*70}")
print("ALL PROCESSING COMPLETE!")
print(f"{'='*70}")
```

    ======================================================================
    HYDRAULIC SYSTEM CONDITION MONITORING - DECISION TREE APPROACH
    ======================================================================
    
    Dataset shape: (2205, 22)
    Number of samples: 2205
    Number of features: 17
    
    
    ######################################################################
    # TRAINING MODEL FOR: Cooler_Condition
    ######################################################################
    
    Class distribution:
    Cooler_Condition
    3      732
    20     732
    100    741
    Name: count, dtype: int64
    
    ──────────────────────────────────────────────────────────────────────
    Training Decision Tree Model...
    ──────────────────────────────────────────────────────────────────────
    ✓ Training completed!
    
    ======================================================================
    RESULTS FOR: Cooler_Condition
    ======================================================================
    
    Training Set Metrics:
      Accuracy:  0.9938
      Precision: 0.9938
      Recall:    0.9938
      F1-Score:  0.9938
    
    Test Set Metrics:
      Accuracy:  0.9841
      Precision: 0.9846
      Recall:    0.9841
      F1-Score:  0.9841
    
    ✓ Good generalization (gap: 0.0096)
    
    ──────────────────────────────────────────────────────────────────────
    Cross-Validation (5-fold):
    ──────────────────────────────────────────────────────────────────────
      CV Accuracy Scores: [0.99433428 0.99716714 0.99150142 0.98866856 0.98579545]
      Mean CV Accuracy: 0.9915 (+/- 0.0040)
    
    ──────────────────────────────────────────────────────────────────────
    Generating Visualizations...
    ──────────────────────────────────────────────────────────────────────
    
    ✓ Confusion matrix saved as 'confusion_matrix_Cooler_Condition.png'
    ✓ Feature importance saved as 'feature_importance_Cooler_Condition.png'
    
    Top 10 Most Important Features:
      1. TS4: 0.5073
      2. PS5: 0.4678
      3. CP: 0.0250
      4. SE: 0.0000
      5. CE: 0.0000
      6. TS3: 0.0000
      7. TS2: 0.0000
      8. TS1: 0.0000
      9. VS1: 0.0000
      10. FS2: 0.0000
    ✓ Decision tree saved as 'decision_tree_Cooler_Condition.png'
    
    ──────────────────────────────────────────────────────────────────────
    Detailed Classification Report:
    ──────────────────────────────────────────────────────────────────────
                  precision    recall  f1-score   support
    
               3       0.96      1.00      0.98       146
              20       0.99      0.96      0.98       147
             100       1.00      0.99      1.00       148
    
        accuracy                           0.98       441
       macro avg       0.98      0.98      0.98       441
    weighted avg       0.98      0.98      0.98       441
    
    
    ──────────────────────────────────────────────────────────────────────
    Model Complexity:
    ──────────────────────────────────────────────────────────────────────
      Tree Depth: 4
      Number of Leaves: 7
      Number of Features Used: 3
    
    
    ######################################################################
    # TRAINING MODEL FOR: Valve_Condition
    ######################################################################
    
    Class distribution:
    Valve_Condition
    73      360
    80      360
    90      360
    100    1125
    Name: count, dtype: int64
    
    ──────────────────────────────────────────────────────────────────────
    Training Decision Tree Model...
    ──────────────────────────────────────────────────────────────────────
    ✓ Training completed!
    
    ======================================================================
    RESULTS FOR: Valve_Condition
    ======================================================================
    
    Training Set Metrics:
      Accuracy:  0.7959
      Precision: 0.8248
      Recall:    0.7959
      F1-Score:  0.8011
    
    Test Set Metrics:
      Accuracy:  0.7279
      Precision: 0.7556
      Recall:    0.7279
      F1-Score:  0.7331
    
    ✓ Good generalization (gap: 0.0680)
    
    ──────────────────────────────────────────────────────────────────────
    Cross-Validation (5-fold):
    ──────────────────────────────────────────────────────────────────────
      CV Accuracy Scores: [0.73654391 0.77053824 0.75354108 0.69971671 0.68465909]
      Mean CV Accuracy: 0.7290 (+/- 0.0323)
    
    ──────────────────────────────────────────────────────────────────────
    Generating Visualizations...
    ──────────────────────────────────────────────────────────────────────
    
    ✓ Confusion matrix saved as 'confusion_matrix_Valve_Condition.png'
    ✓ Feature importance saved as 'feature_importance_Valve_Condition.png'
    
    Top 10 Most Important Features:
      1. SE: 0.3217
      2. PS2: 0.2538
      3. PS1: 0.0723
      4. TS1: 0.0650
      5. FS1: 0.0527
      6. PS6: 0.0393
      7. VS1: 0.0336
      8. TS2: 0.0319
      9. EPS1: 0.0305
      10. TS4: 0.0291
    ✓ Decision tree saved as 'decision_tree_Valve_Condition.png'
    
    ──────────────────────────────────────────────────────────────────────
    Detailed Classification Report:
    ──────────────────────────────────────────────────────────────────────
                  precision    recall  f1-score   support
    
              73       0.54      0.86      0.67        72
              80       0.53      0.51      0.52        72
              90       0.59      0.50      0.54        72
             100       0.95      0.83      0.88       225
    
        accuracy                           0.73       441
       macro avg       0.65      0.68      0.65       441
    weighted avg       0.76      0.73      0.73       441
    
    
    ──────────────────────────────────────────────────────────────────────
    Model Complexity:
    ──────────────────────────────────────────────────────────────────────
      Tree Depth: 10
      Number of Leaves: 57
      Number of Features Used: 16
    
    
    ######################################################################
    # TRAINING MODEL FOR: Internal_Pump_Leakage
    ######################################################################
    
    Class distribution:
    Internal_Pump_Leakage
    0    1221
    1     492
    2     492
    Name: count, dtype: int64
    
    ──────────────────────────────────────────────────────────────────────
    Training Decision Tree Model...
    ──────────────────────────────────────────────────────────────────────
    ✓ Training completed!
    
    ======================================================================
    RESULTS FOR: Internal_Pump_Leakage
    ======================================================================
    
    Training Set Metrics:
      Accuracy:  0.9807
      Precision: 0.9807
      Recall:    0.9807
      F1-Score:  0.9807
    
    Test Set Metrics:
      Accuracy:  0.9705
      Precision: 0.9717
      Recall:    0.9705
      F1-Score:  0.9706
    
    ✓ Good generalization (gap: 0.0102)
    
    ──────────────────────────────────────────────────────────────────────
    Cross-Validation (5-fold):
    ──────────────────────────────────────────────────────────────────────
      CV Accuracy Scores: [0.9631728  0.9631728  0.97167139 0.96033994 0.94602273]
      Mean CV Accuracy: 0.9609 (+/- 0.0083)
    
    ──────────────────────────────────────────────────────────────────────
    Generating Visualizations...
    ──────────────────────────────────────────────────────────────────────
    
    ✓ Confusion matrix saved as 'confusion_matrix_Internal_Pump_Leakage.png'
    ✓ Feature importance saved as 'feature_importance_Internal_Pump_Leakage.png'
    
    Top 10 Most Important Features:
      1. SE: 0.7267
      2. TS2: 0.0938
      3. FS1: 0.0620
      4. EPS1: 0.0304
      5. TS3: 0.0265
      6. VS1: 0.0199
      7. PS1: 0.0194
      8. PS6: 0.0147
      9. FS2: 0.0048
      10. TS1: 0.0009
    ✓ Decision tree saved as 'decision_tree_Internal_Pump_Leakage.png'
    
    ──────────────────────────────────────────────────────────────────────
    Detailed Classification Report:
    ──────────────────────────────────────────────────────────────────────
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       244
               1       0.90      0.97      0.94        98
               2       0.98      0.91      0.94        99
    
        accuracy                           0.97       441
       macro avg       0.96      0.96      0.96       441
    weighted avg       0.97      0.97      0.97       441
    
    
    ──────────────────────────────────────────────────────────────────────
    Model Complexity:
    ──────────────────────────────────────────────────────────────────────
      Tree Depth: 8
      Number of Leaves: 24
      Number of Features Used: 12
    
    
    ######################################################################
    # TRAINING MODEL FOR: Hydraulic_Accumulator
    ######################################################################
    
    Class distribution:
    Hydraulic_Accumulator
    90     808
    100    399
    115    399
    130    599
    Name: count, dtype: int64
    
    ──────────────────────────────────────────────────────────────────────
    Training Decision Tree Model...
    ──────────────────────────────────────────────────────────────────────
    ✓ Training completed!
    
    ======================================================================
    RESULTS FOR: Hydraulic_Accumulator
    ======================================================================
    
    Training Set Metrics:
      Accuracy:  0.9155
      Precision: 0.9179
      Recall:    0.9155
      F1-Score:  0.9160
    
    Test Set Metrics:
      Accuracy:  0.8980
      Precision: 0.8981
      Recall:    0.8980
      F1-Score:  0.8979
    
    ✓ Good generalization (gap: 0.0176)
    
    ──────────────────────────────────────────────────────────────────────
    Cross-Validation (5-fold):
    ──────────────────────────────────────────────────────────────────────
      CV Accuracy Scores: [0.87818697 0.82152975 0.85835694 0.84419263 0.82954545]
      Mean CV Accuracy: 0.8464 (+/- 0.0203)
    
    ──────────────────────────────────────────────────────────────────────
    Generating Visualizations...
    ──────────────────────────────────────────────────────────────────────
    
    ✓ Confusion matrix saved as 'confusion_matrix_Hydraulic_Accumulator.png'
    ✓ Feature importance saved as 'feature_importance_Hydraulic_Accumulator.png'
    
    Top 10 Most Important Features:
      1. SE: 0.1977
      2. TS4: 0.1976
      3. FS2: 0.1316
      4. TS2: 0.0997
      5. EPS1: 0.0715
      6. TS1: 0.0683
      7. PS1: 0.0626
      8. CE: 0.0477
      9. PS3: 0.0413
      10. CP: 0.0318
    ✓ Decision tree saved as 'decision_tree_Hydraulic_Accumulator.png'
    
    ──────────────────────────────────────────────────────────────────────
    Detailed Classification Report:
    ──────────────────────────────────────────────────────────────────────
                  precision    recall  f1-score   support
    
              90       0.92      0.91      0.92       161
             100       0.88      0.93      0.90        80
             115       0.86      0.84      0.85        80
             130       0.90      0.91      0.90       120
    
        accuracy                           0.90       441
       macro avg       0.89      0.89      0.89       441
    weighted avg       0.90      0.90      0.90       441
    
    
    ──────────────────────────────────────────────────────────────────────
    Model Complexity:
    ──────────────────────────────────────────────────────────────────────
      Tree Depth: 10
      Number of Leaves: 55
      Number of Features Used: 15
    
    
    ######################################################################
    # TRAINING MODEL FOR: Stable_Flag
    ######################################################################
    
    Class distribution:
    Stable_Flag
    0    1449
    1     756
    Name: count, dtype: int64
    
    ──────────────────────────────────────────────────────────────────────
    Training Decision Tree Model...
    ──────────────────────────────────────────────────────────────────────
    ✓ Training completed!
    
    ======================================================================
    RESULTS FOR: Stable_Flag
    ======================================================================
    
    Training Set Metrics:
      Accuracy:  0.9620
      Precision: 0.9624
      Recall:    0.9620
      F1-Score:  0.9617
    
    Test Set Metrics:
      Accuracy:  0.9388
      Precision: 0.9421
      Recall:    0.9388
      F1-Score:  0.9374
    
    ✓ Good generalization (gap: 0.0232)
    
    ──────────────────────────────────────────────────────────────────────
    Cross-Validation (5-fold):
    ──────────────────────────────────────────────────────────────────────
      CV Accuracy Scores: [0.9490085  0.93767705 0.95750708 0.92351275 0.95454545]
      Mean CV Accuracy: 0.9445 (+/- 0.0125)
    
    ──────────────────────────────────────────────────────────────────────
    Generating Visualizations...
    ──────────────────────────────────────────────────────────────────────
    
    ✓ Confusion matrix saved as 'confusion_matrix_Stable_Flag.png'
    ✓ Feature importance saved as 'feature_importance_Stable_Flag.png'
    
    Top 10 Most Important Features:
      1. SE: 0.4622
      2. TS4: 0.1645
      3. TS2: 0.1245
      4. TS3: 0.1100
      5. PS1: 0.0644
      6. PS2: 0.0261
      7. PS6: 0.0134
      8. CP: 0.0130
      9. PS3: 0.0083
      10. PS5: 0.0051
    ✓ Decision tree saved as 'decision_tree_Stable_Flag.png'
    
    ──────────────────────────────────────────────────────────────────────
    Detailed Classification Report:
    ──────────────────────────────────────────────────────────────────────
                  precision    recall  f1-score   support
    
               0       0.92      0.99      0.96       290
               1       0.98      0.83      0.90       151
    
        accuracy                           0.94       441
       macro avg       0.95      0.91      0.93       441
    weighted avg       0.94      0.94      0.94       441
    
    
    ──────────────────────────────────────────────────────────────────────
    Model Complexity:
    ──────────────────────────────────────────────────────────────────────
      Tree Depth: 10
      Number of Leaves: 40
      Number of Features Used: 15
    
    
    ======================================================================
    OVERALL SUMMARY - ALL TARGETS
    ======================================================================
    
                          Train Accuracy Test Accuracy Train Precision  \
    Cooler_Condition              0.9938        0.9841          0.9938   
    Valve_Condition               0.7959        0.7279          0.8248   
    Internal_Pump_Leakage         0.9807        0.9705          0.9807   
    Hydraulic_Accumulator         0.9155        0.8980          0.9179   
    Stable_Flag                   0.9620        0.9388          0.9624   
    
                          Test Precision  
    Cooler_Condition              0.9846  
    Valve_Condition               0.7556  
    Internal_Pump_Leakage         0.9717  
    Hydraulic_Accumulator         0.8981  
    Stable_Flag                   0.9421  
    
    ──────────────────────────────────────────────────────────────────────
    Best Performing Model: Cooler_Condition (0.9841)
    Worst Performing Model: Valve_Condition (0.7279)
    
    ──────────────────────────────────────────────────────────────────────
    Saving Models...
    ──────────────────────────────────────────────────────────────────────
    ✓ Model saved: dt_model_Cooler_Condition.pkl
    ✓ Model saved: dt_model_Valve_Condition.pkl
    ✓ Model saved: dt_model_Internal_Pump_Leakage.pkl
    ✓ Model saved: dt_model_Hydraulic_Accumulator.pkl
    ✓ Model saved: dt_model_Stable_Flag.pkl
    
    ======================================================================
    ALL PROCESSING COMPLETE!
    ======================================================================
    


    
![png](output_0_1.png)
    



    
![png](output_0_2.png)
    



    
![png](output_0_3.png)
    



    
![png](output_0_4.png)
    



    
![png](output_0_5.png)
    



    
![png](output_0_6.png)
    



    
![png](output_0_7.png)
    



    
![png](output_0_8.png)
    



    
![png](output_0_9.png)
    



    
![png](output_0_10.png)
    



    
![png](output_0_11.png)
    



    
![png](output_0_12.png)
    



    
![png](output_0_13.png)
    



    
![png](output_0_14.png)
    



    
![png](output_0_15.png)
    



```python

```


```python

```
