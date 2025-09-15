"""
This script operationalizes the complete model training, validation, and
serialization pipeline for the driver risk assessment task. It trains a regularized
XGBoost classifier on the engineered feature set and employs stratified k-fold cross-validation
to ensure the model's generalization capabilities. Critically, it moves beyond standard metrics
by programmatically optimizing the classification threshold to maximize precision, aligning the
 model's performance with key business objectives, before serializing the final model and its
 evaluation results for deployment.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, \
    f1_score, accuracy_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


class IncentiveBasedRiskModel:
    """XGBoost model for predicting discount eligibility in telematics insurance"""

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=3.0,
            reg_lambda=4.0,
            min_child_weight=10,
            gamma=0.1,
            random_state=42,
            verbosity=0,
            eval_metric='logloss',
            objective='binary:logistic',
            use_label_encoder=False
        )

        self.feature_names = None
        self.is_trained = False
        self.optimal_threshold = 0.5

    def load_ml_dataset(self, data_path="data/ml_ready_dataset.csv"):
        """Load the ML-ready dataset"""
        if not os.path.exists(data_path):
            print(f" ML dataset not found at {data_path}")
            print("   Run 'python src/feature_engineering.py' first!")
            return None, None

        df = pd.read_csv(data_path)


        X = df.drop('discount_eligible', axis=1)
        y = df['discount_eligible'].values

        self.feature_names = list(X.columns)

        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Discount eligible: {sum(y)} ({sum(y) / len(y) * 100:.1f}%)")
        print(f"   Standard rate: {len(y) - sum(y)} ({(len(y) - sum(y)) / len(y) * 100:.1f}%)")

        return X, y

    def find_optimal_precision_threshold(self, X_test, y_test):
        """Find the threshold that maximizes precision"""
        probas = self.model.predict_proba(X_test)[:, 1]

        best_threshold = 0.5
        best_precision = 0

        print("Finding optimal threshold for precision...")


        for threshold in np.arange(0.3, 0.9, 0.05):
            y_pred_thresh = (probas >= threshold).astype(int)

            if sum(y_pred_thresh) > 0:
                precision = precision_score(y_test, y_pred_thresh)
                if precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold

        print(f"   Optimal threshold for precision: {best_threshold:.2f}")
        print(f"   Precision at optimal threshold: {best_precision:.3f}")

        return best_threshold, best_precision

    def evaluate_with_threshold_optimization(self, X_test, y_test):
        """Evaluate model with optimized threshold for precision"""

        optimal_threshold, optimal_precision = self.find_optimal_precision_threshold(X_test, y_test)
        self.optimal_threshold = optimal_threshold

        probas = self.model.predict_proba(X_test)[:, 1]
        y_pred_optimized = (probas >= optimal_threshold).astype(int)

        precision_opt = precision_score(y_test, y_pred_optimized)
        recall_opt = recall_score(y_test, y_pred_optimized)
        f1_opt = f1_score(y_test, y_pred_optimized)
        accuracy_opt = accuracy_score(y_test, y_pred_optimized)

        print(f"\n OPTIMIZED PERFORMANCE (Threshold: {optimal_threshold:.2f}):")
        print(f"   Precision: {precision_opt:.3f} (Target: >0.90)")
        print(f"   Recall: {recall_opt:.3f}")
        print(f"   F1 Score: {f1_opt:.3f}")
        print(f"   Accuracy: {accuracy_opt:.3f}")


        cm_opt = confusion_matrix(y_test, y_pred_optimized)
        print(f"\nOptimized Confusion Matrix:")
        print(f"   True Negatives (Correct Standard): {cm_opt[0, 0]}")
        print(f"   False Positives (Wrong Discount): {cm_opt[0, 1]}")
        print(f"   False Negatives (Missed Discount): {cm_opt[1, 0]}")
        print(f"   True Positives (Correct Discount): {cm_opt[1, 1]}")

        return {
            'threshold': optimal_threshold,
            'precision': precision_opt,
            'recall': recall_opt,
            'f1': f1_opt,
            'accuracy': accuracy_opt,
            'y_pred_optimized': y_pred_optimized
        }

    def train_model(self, X, y, test_size=0.25):
        """Train the XGBoost model optimized for larger dataset"""
        print("Training BALANCED XGBoost model for larger dataset...")


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")


        self.model.fit(X_train, y_train)
        self.is_trained = True

        print("Performing cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')

        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        overfitting_gap = train_acc - test_acc

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]


        auc_score = roc_auc_score(y_test, y_pred_proba)

        print("\nBALANCED MODEL PERFORMANCE:")
        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
        print(f"Overfitting Gap: {overfitting_gap:.3f}")
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"AUC Score: {auc_score:.3f}")

        if overfitting_gap <= 0.08:
            print(f"OVERFITTING TARGET MET: {overfitting_gap:.1%} <= 8%")
        else:
            print(f"⚠OVERFITTING: {overfitting_gap:.1%}")

        if test_acc >= 0.78:
            print(f"ACCURACY TARGET MET: {test_acc:.1%} >= 78%")
        elif test_acc >= 0.75:
            print(f"ACCEPTABLE ACCURACY: {test_acc:.1%}")
        else:
            print(f"LOW ACCURACY: {test_acc:.1%}")

        print("\nStandard Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['Standard Rate', 'Discount Eligible']))

        optimized_results = self.evaluate_with_threshold_optimization(X_test, y_test)

        self.analyze_business_impact(y_test, optimized_results['y_pred_optimized'], y_pred_proba)

        self.show_feature_importance()


        model_path = "models/discount_eligibility_model.pkl"
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"\nModel saved to '{model_path}'")


        results = {
            'model': self.model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'overfitting_gap': overfitting_gap,
            'cv_scores': cv_scores,
            'auc_score': auc_score,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'optimized_results': optimized_results
        }

        results_path = "models/discount_eligibility_results.pkl"
        joblib.dump(results, results_path)
        print(f"Results saved to '{results_path}'")

        return results

    def analyze_business_impact(self, y_true, y_pred, y_pred_proba):
        """Analyze business impact of the model"""
        print("\n💰 BUSINESS IMPACT ANALYSIS:")

        base_premium = 1100
        discount_rates = {0: 0.0, 1: 0.10, 2: 0.20, 3: 0.30}

        correct_discounts = sum((y_true == 1) & (y_pred == 1))
        missed_discounts = sum((y_true == 1) & (y_pred == 0))
        wrong_discounts = sum((y_true == 0) & (y_pred == 1))

        avg_discount = 0.15
        avg_savings_per_driver = base_premium * avg_discount

        total_correct_savings = correct_discounts * avg_savings_per_driver
        total_missed_savings = missed_discounts * avg_savings_per_driver
        total_wrong_costs = wrong_discounts * avg_savings_per_driver

        print(f"   Correctly identified discount drivers: {correct_discounts}")
        print(f"   Missed discount opportunities: {missed_discounts}")
        print(f"   Incorrectly given discounts: {wrong_discounts}")
        print(f"   Total savings delivered: ${total_correct_savings:,.0f}")
        print(f"   Missed savings opportunity: ${total_missed_savings:,.0f}")
        print(f"   Cost of wrong discounts: ${total_wrong_costs:,.0f}")


        precision = correct_discounts / (correct_discounts + wrong_discounts) if (
                                                                                         correct_discounts + wrong_discounts) > 0 else 0
        recall = correct_discounts / (correct_discounts + missed_discounts) if (
                                                                                       correct_discounts + missed_discounts) > 0 else 0

        print(f"   Discount precision: {precision:.3f} (accuracy of discount awards)")
        print(f"   Discount recall: {recall:.3f} (% of eligible drivers identified)")

    def show_feature_importance(self):
        """Display feature importance"""
        if not self.is_trained:
            print("Model not trained yet!")
            return

        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print("\nTOP 10 MOST IMPORTANT FEATURES:")
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")

    def predict_discount_eligibility(self, driver_features, use_optimal_threshold=True):
        """Predict discount eligibility for new drivers"""
        if not self.is_trained:
            print("Model not trained yet!")
            return None

        probability = self.model.predict_proba(driver_features)[0, 1]

        threshold = self.optimal_threshold if use_optimal_threshold else 0.5
        prediction = 1 if probability >= threshold else 0

        base_premium = 1100
        if prediction == 1:
            if probability >= 0.8:
                discount_tier = 3
                discount_rate = 0.30
            elif probability >= 0.6:
                discount_tier = 2
                discount_rate = 0.20
            else:
                discount_tier = 1
                discount_rate = 0.10

            final_premium = base_premium * (1 - discount_rate)
            annual_savings = base_premium - final_premium

            return {
                'discount_eligible': True,
                'discount_tier': discount_tier,
                'discount_rate': discount_rate,
                'probability': probability,
                'threshold_used': threshold,
                'estimated_premium': final_premium,
                'annual_savings': annual_savings
            }
        else:
            return {
                'discount_eligible': False,
                'discount_tier': 0,
                'discount_rate': 0.0,
                'probability': probability,
                'threshold_used': threshold,
                'estimated_premium': base_premium,
                'annual_savings': 0
            }

    def generate_sample_predictions(self, X_test, n_samples=5):
        """Generate sample predictions for demonstration"""
        print("\nSAMPLE DISCOUNT PREDICTIONS (With Optimized Threshold):")

        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

        for i, idx in enumerate(sample_indices):
            driver_features = X_test.iloc[idx:idx + 1]
            result = self.predict_discount_eligibility(driver_features)

            print(f"\n   Driver {i + 1}:")
            print(f"   Discount Eligible: {'Yes' if result['discount_eligible'] else 'No'}")
            print(f"   Confidence: {result['probability']:.3f}")
            print(f"   Threshold Used: {result['threshold_used']:.3f}")
            print(f"   Estimated Premium: ${result['estimated_premium']:.0f}")
            print(f"   Annual Savings: ${result['annual_savings']:.0f}")
            if result['discount_eligible']:
                print(f"   Discount Tier: {result['discount_tier']} ({result['discount_rate']:.0%} off)")


def main():
    """Main training pipeline"""
    print(" Starting BALANCED Telematics Model Training...")

    model = IncentiveBasedRiskModel()

    X, y = model.load_ml_dataset()

    if X is None:
        return

    results = model.train_model(X, y)

    model.generate_sample_predictions(results['X_test'])

    print("\nBalanced Model training complete!")
    print(f"\nFINAL RESULTS SUMMARY:")
    print(f"   Test Accuracy: {results['test_accuracy']:.1%}")
    print(f"   Overfitting Gap: {results['overfitting_gap']:.1%}")
    print(f"   Cross-Validation: {results['cv_scores'].mean():.1%} ± {results['cv_scores'].std():.1%}")

    print("\nNext steps:")
    print("1. Run 'python src/api.py' to start the API")
    print("2. Run 'streamlit run src/dashboard.py' to start the dashboard")

    return model, results


if __name__ == "__main__":
    trained_model, training_results = main()