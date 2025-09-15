import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier


from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix,
                             roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import time
import joblib
import os


class TelematicsModelComparison:
    """Comprehensive comparison of ML models for telematics discount prediction"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_data(self, data_path="data/ml_ready_dataset.csv"):
        """Load and prepare the ML dataset"""
        print("Loading ML dataset...")

        if not os.path.exists(data_path):
            print(f"Dataset not found at {data_path}")
            print("   Run 'python src/feature_engineering.py' first!")
            return False

        df = pd.read_csv(data_path)

        X = df.drop('discount_eligible', axis=1)
        y = df['discount_eligible'].values


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )


        X_scaled = self.scaler.fit_transform(X)
        self.X_train_scaled, self.X_test_scaled, _, _ = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )

        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Training set: {len(self.X_train)} samples")
        print(f"   Test set: {len(self.X_test)} samples")
        print(f"   Class distribution: {np.bincount(y)}")

        return True

    def initialize_models(self):
        """Initialize all models for comparison"""
        print("Initializing models...")

        self.models = {
            'XGBoost_Conservative': xgb.XGBClassifier(
                n_estimators=18, max_depth=4, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=4.0, reg_lambda=4.0, min_child_weight=10,
                gamma=0.1, random_state=42, verbosity=0
            ),
            'XGBoost_Aggressive': xgb.XGBClassifier(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                random_state=42, verbosity=0
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=50, max_depth=6, min_samples_split=10,
                min_samples_leaf=5, random_state=42
            ),
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.05,
                random_state=42
            ),
            'Extra_Trees': ExtraTreesClassifier(
                n_estimators=50, max_depth=6, min_samples_split=10,
                random_state=42
            ),
            'Decision_Tree': DecisionTreeClassifier(
                max_depth=6, min_samples_split=10, min_samples_leaf=5,
                random_state=42
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=30, learning_rate=0.8, random_state=42
            ),

            'Logistic_Regression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=42
            ),
            'SVM_RBF': SVC(
                C=1.0, kernel='rbf', probability=True, random_state=42
            ),
            'SVM_Linear': SVC(
                C=1.0, kernel='linear', probability=True, random_state=42
            ),
            'Neural_Network_Small': MLPClassifier(
                hidden_layer_sizes=(50,), max_iter=500,
                alpha=0.1, random_state=42
            ),
            'Neural_Network_Deep': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500,
                alpha=0.01, random_state=42
            ),
            'Naive_Bayes': GaussianNB()
        }

        print(f"{len(self.models)} models initialized")

    def train_and_evaluate_model(self, name, model, use_scaled=False):
        """Train and evaluate a single model"""
        print(f"   Training {name}...")

        X_train = self.X_train_scaled if use_scaled else self.X_train
        X_test = self.X_test_scaled if use_scaled else self.X_test


        start_time = time.time()
        model.fit(X_train, self.y_train)
        training_time = time.time() - start_time


        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        prediction_time = time.time() - start_time


        train_acc = model.score(X_train, self.y_train)
        test_acc = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)


        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, self.y_train, cv=cv, scoring='accuracy')


        auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None

        return {
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'overfitting': train_acc - test_acc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    def run_comparison(self):
        """Run comprehensive model comparison"""
        print("Starting comprehensive model comparison...")

        scaled_models = ['Logistic_Regression', 'SVM_RBF', 'SVM_Linear',
                         'Neural_Network_Small', 'Neural_Network_Deep', 'Naive_Bayes']

        for name, model in self.models.items():
            try:
                use_scaled = name in scaled_models
                result = self.train_and_evaluate_model(name, model, use_scaled)
                self.results[name] = result

                print(f"      {name}: Test Acc = {result['test_accuracy']:.3f}, "
                      f"CV = {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")

            except Exception as e:
                print(f"     {name} failed: {str(e)}")
                self.results[name] = None

        print(" Model comparison complete!")

    def create_results_dataframe(self):
        """Create a comprehensive results DataFrame"""
        data = []

        for name, result in self.results.items():
            if result is not None:
                data.append({
                    'Model': name.replace('_', ' '),
                    'Test_Accuracy': result['test_accuracy'],
                    'Train_Accuracy': result['train_accuracy'],
                    'CV_Mean': result['cv_mean'],
                    'CV_Std': result['cv_std'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1_Score': result['f1_score'],
                    'AUC_Score': result['auc_score'],
                    'Overfitting': result['overfitting'],
                    'Training_Time': result['training_time'],
                    'Prediction_Time': result['prediction_time']
                })

        return pd.DataFrame(data).sort_values('Test_Accuracy', ascending=False)

    def print_detailed_results(self):
        """Print detailed results summary"""
        results_df = self.create_results_dataframe()

        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL COMPARISON RESULTS")
        print("=" * 80)

        print("\nTOP 5 MODELS BY TEST ACCURACY:")
        top_5 = results_df.head(5)
        for i, row in top_5.iterrows():
            print(f"{i + 1}. {row['Model']}: {row['Test_Accuracy']:.1%} "
                  f"(CV: {row['CV_Mean']:.1%} ± {row['CV_Std']:.1%})")

        print("\nACCURACY COMPARISON:")
        print(f"{'Model':<20} {'Test':<8} {'Train':<8} {'CV':<12} {'Overfit':<8}")
        print("-" * 60)
        for _, row in results_df.iterrows():
            print(f"{row['Model']:<20} {row['Test_Accuracy']:.3f}   "
                  f"{row['Train_Accuracy']:.3f}   {row['CV_Mean']:.3f}±{row['CV_Std']:.3f}   "
                  f"{row['Overfitting']:.3f}")

        print("\nPERFORMANCE METRICS:")
        print(f"{'Model':<20} {'Precision':<10} {'Recall':<8} {'F1':<8} {'AUC':<8}")
        print("-" * 60)
        for _, row in results_df.iterrows():
            auc_str = f"{row['AUC_Score']:.3f}" if pd.notna(row['AUC_Score']) else "N/A"
            print(f"{row['Model']:<20} {row['Precision']:.3f}     "
                  f"{row['Recall']:.3f}   {row['F1_Score']:.3f}   {auc_str}")

        print("\nTIMING ANALYSIS:")
        print(f"{'Model':<20} {'Train_Time':<12} {'Predict_Time':<12}")
        print("-" * 50)
        for _, row in results_df.iterrows():
            print(f"{row['Model']:<20} {row['Training_Time']:.3f}s       "
                  f"{row['Prediction_Time']:.4f}s")

        print("\n BUSINESS IMPACT ANALYSIS:")
        best_model = results_df.iloc[0]
        print(f"Best Model: {best_model['Model']}")
        print(f"Expected Accuracy: {best_model['Test_Accuracy']:.1%}")
        print(f"Overfitting Level: {best_model['Overfitting']:.3f} "
              f"({'Low' if best_model['Overfitting'] < 0.05 else 'High'})")

        return results_df

    def plot_model_comparison(self):
        """Create comprehensive visualization of model performance"""
        results_df = self.create_results_dataframe()


        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Test Accuracy Comparison', 'Overfitting Analysis',
                            'Precision vs Recall', 'Training Time vs Accuracy'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        fig.add_trace(
            go.Bar(x=results_df['Model'], y=results_df['Test_Accuracy'],
                   name='Test Accuracy', marker_color='steelblue'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=results_df['Train_Accuracy'], y=results_df['Test_Accuracy'],
                       mode='markers+text', text=results_df['Model'],
                       textposition="top center", name='Train vs Test',
                       marker=dict(size=10, color='red')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[0.5, 1.0], y=[0.5, 1.0], mode='lines',
                       name='Perfect Fit', line=dict(dash='dash', color='gray')),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=results_df['Recall'], y=results_df['Precision'],
                       mode='markers+text', text=results_df['Model'],
                       textposition="top center", name='Precision vs Recall',
                       marker=dict(size=10, color='green')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=results_df['Training_Time'], y=results_df['Test_Accuracy'],
                       mode='markers+text', text=results_df['Model'],
                       textposition="top center", name='Time vs Accuracy',
                       marker=dict(size=10, color='purple')),
            row=2, col=2
        )


        fig.update_layout(
            title_text="Telematics Model Comparison Dashboard",
            height=800,
            showlegend=False
        )


        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Train Accuracy", row=1, col=2)
        fig.update_yaxes(title_text="Test Accuracy", row=1, col=2)
        fig.update_xaxes(title_text="Recall", row=2, col=1)
        fig.update_yaxes(title_text="Precision", row=2, col=1)
        fig.update_xaxes(title_text="Training Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Test Accuracy", row=2, col=2)

        fig.show()

        return fig

    def save_results(self):
        """Save results and best model"""
        results_df = self.create_results_dataframe()


        results_df.to_csv('data/model_comparison_results.csv', index=False)
        print("Results saved to 'data/model_comparison_results.csv'")

        best_model_name = results_df.iloc[0]['Model'].replace(' ', '_')
        best_model = self.results[best_model_name]['model']

        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, f'models/best_model_{best_model_name.lower()}.pkl')
        print(f"Best model ({best_model_name}) saved to 'models/best_model_{best_model_name.lower()}.pkl'")

        return results_df


def main():
    """Main comparison pipeline"""
    print("Starting Comprehensive Telematics Model Comparison...")

    comparison = TelematicsModelComparison()


    if not comparison.load_data():
        return

    comparison.initialize_models()

    comparison.run_comparison()

    results_df = comparison.print_detailed_results()

    print("\nCreating comparison visualizations...")
    comparison.plot_model_comparison()

    comparison.save_results()

    print("\nComprehensive model comparison complete!")
    print("\nKey Findings:")
    best_model = results_df.iloc[0]
    print(f"Best Model: {best_model['Model']}")
    print(f"Best Accuracy: {best_model['Test_Accuracy']:.1%}")
    print(f"Cross-Validation: {best_model['CV_Mean']:.1%} ± {best_model['CV_Std']:.1%}")



if __name__ == "__main__":
    main()