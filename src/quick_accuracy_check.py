import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os


def quick_accuracy_check():
    """Quick check of current XGBoost model accuracy"""

    print("Quick Accuracy Check for Telematics Model")
    print("=" * 50)

    # Load data
    data_path = "data/ml_ready_dataset.csv"
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("   Run the pipeline first:")
        print("   1. python data_simulator.py")
        print("   2. python src/feature_engineering.py")
        return

    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} samples")

    X = df.drop('discount_eligible', axis=1)
    y = df['discount_eligible'].values

    print(f"   Features: {X.shape[1]}")
    print(f"   Discount eligible: {sum(y)} ({sum(y) / len(y) * 100:.1f}%)")
    print(f"   Standard rate: {len(y) - sum(y)} ({(len(y) - sum(y)) / len(y) * 100:.1f}%)")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\nTrain/Test Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")


    print(f"\nTraining Conservative XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=18,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=4.0,
        reg_lambda=4.0,
        min_child_weight=10,
        gamma=0.1,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train)


    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')


    y_pred = model.predict(X_test)

    print(f"\nACCURACY RESULTS:")
    print(f"   Training Accuracy: {train_acc:.3f} ({train_acc:.1%})")
    print(f"   Test Accuracy: {test_acc:.3f} ({test_acc:.1%})")
    print(f"   Cross-Validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"   Overfitting Gap: {train_acc - test_acc:.3f}")


    if test_acc >= 0.70 and test_acc <= 0.78:
        status = "PERFECT RANGE"
        color = "green"
    elif test_acc > 0.85:
        status = "TOO HIGH (Overfitting suspected)"
        color = "red"
    elif test_acc < 0.65:
        status = "TOO LOW (Underfitting)"
        color = "orange"
    else:
        status = "ACCEPTABLE RANGE"
        color = "blue"

    print(f"\nStatus: {status}")
    print(f"   Target Range: 70-78%")
    print(f"   Your Result: {test_acc:.1%}")


    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"   True Negatives (Correct Standard): {cm[0, 0]}")
    print(f"   False Positives (Wrong Discount): {cm[0, 1]}")
    print(f"   False Negatives (Missed Discount): {cm[1, 0]}")
    print(f"   True Positives (Correct Discount): {cm[1, 1]}")


    correct_discounts = cm[1, 1]
    missed_discounts = cm[1, 0]
    wrong_discounts = cm[0, 1]

    print(f"\nBusiness Impact:")
    print(f"   Correctly awarded discounts: {correct_discounts}")
    print(f"   Missed discount opportunities: {missed_discounts}")
    print(f"   Incorrectly awarded discounts: {wrong_discounts}")

    precision = correct_discounts / (correct_discounts + wrong_discounts) if (
                                                                                         correct_discounts + wrong_discounts) > 0 else 0
    recall = correct_discounts / (correct_discounts + missed_discounts) if (
                                                                                       correct_discounts + missed_discounts) > 0 else 0

    print(f"   Discount Precision: {precision:.1%} (accuracy of discount awards)")
    print(f"   Discount Recall: {recall:.1%} (% of eligible drivers found)")


    feature_importance = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print(f"\nTop 5 Most Important Features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"   {i + 1}. {row['feature']}: {row['importance']:.3f}")

    print(f"\n{'=' * 50}")



    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'status': status
    }


if __name__ == "__main__":
    results = quick_accuracy_check()