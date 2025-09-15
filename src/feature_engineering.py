"""
This script serves as the project's data processing and feature engineering pipeline.
It transforms raw, high-frequency time-series data into a flattened, feature-rich dataset
 where each record corresponds to a unique driver. The process involves aggregating raw telematics
 signals into insightful metrics, engineering composite safety scores, and operationalizing business
 rules to create the binary target variable for classification. To enhance model robustness,
 the script injects realistic noise, synthesizes edge cases, and applies the SMOTE-ENN technique
 to rectify class imbalance, ultimately producing a cleansed and balanced dataset primed for model training.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from collections import Counter


class TelematicsFeatureEngineer:
    """Feature engineering for incentive-based telematics insurance"""

    def __init__(self):
        self.scaler = StandardScaler()

    def create_features_from_raw_data(self, df):
        """Convert raw telematics data into ML features for discount prediction"""
        driver_features = []

        for driver_id in df['driver_id'].unique():
            driver_data = df[df['driver_id'] == driver_id]

            avg_speed = driver_data['speed_kph'].mean()
            max_speed = driver_data['speed_kph'].max()
            speed_std = driver_data['speed_kph'].std()

            harsh_braking_count = driver_data['harsh_braking'].sum()
            harsh_accel_count = driver_data['harsh_acceleration'].sum()

            speed_safety_score = max(0, 100 - max(0, avg_speed - 60) * 2)
            braking_safety_score = max(0, 100 - harsh_braking_count * 3)
            accel_safety_score = max(0, 100 - harsh_accel_count * 3)
            consistency_score = max(0, 100 - speed_std * 2)

            overall_safety_score = (speed_safety_score + braking_safety_score +
                                    accel_safety_score + consistency_score) / 4

            features = {
                'driver_id': driver_id,
                'avg_speed_kph': avg_speed,
                'max_speed_kph': max_speed,
                'speed_std': speed_std,
                'speed_safety_score': speed_safety_score,
                'harsh_braking_events': harsh_braking_count,
                'harsh_acceleration_events': harsh_accel_count,
                'braking_safety_score': braking_safety_score,
                'acceleration_safety_score': accel_safety_score,
                'consistency_score': consistency_score,
                'overall_safety_score': overall_safety_score,
                'avg_acceleration': driver_data['acceleration_ms2'].mean(),
                'max_acceleration': driver_data['acceleration_ms2'].max(),
                'min_acceleration': driver_data['acceleration_ms2'].min(),
                'total_distance_km': len(driver_data) * avg_speed / 600,
                'total_trips': driver_data['trip_id'].nunique(),
                'avg_trip_duration': len(driver_data) / driver_data['trip_id'].nunique(),
                'speeding_events': (driver_data['speed_kph'] > 80).sum(),
                'night_driving_ratio': np.random.beta(2, 8),
                'weekend_driving_ratio': np.random.beta(3, 7),
                'phone_usage_events': np.random.poisson(0.5),
                'harsh_cornering_events': np.random.poisson(0.3),
                'driver_profile': driver_data['driver_profile'].iloc[0]
            }

            driver_features.append(features)

        return pd.DataFrame(driver_features)

    def determine_discount_eligibility(self, features_df):
        """Determine discount eligibility based on safety criteria"""

        discount_eligible = []
        discount_tier = []

        for _, driver in features_df.iterrows():
            safety_score = driver['overall_safety_score']
            harsh_events = driver['harsh_braking_events'] + driver['harsh_acceleration_events']
            avg_speed = driver['avg_speed_kph']
            phone_usage = driver['phone_usage_events']

            if (safety_score >= 75 and
                    harsh_events <= 5 and
                    avg_speed <= 70 and
                    phone_usage <= 1):

                if safety_score >= 85:
                    tier = 3
                elif safety_score >= 80:
                    tier = 2
                else:
                    tier = 1

                eligible = True

            elif (safety_score >= 65 and
                  harsh_events <= 8 and
                  avg_speed <= 75):

                tier = 1
                eligible = True

            else:

                tier = 0
                eligible = False

            discount_eligible.append(eligible)
            discount_tier.append(tier)

        features_df['discount_eligible'] = discount_eligible
        features_df['discount_tier'] = discount_tier

        return features_df

    def calculate_premium_examples(self, features_df):
        """Calculate example premiums for each driver"""
        base_premium = 1100
        discount_rates = {0: 0.0, 1: 0.10, 2: 0.20, 3: 0.30}

        premiums = []
        savings_list = []

        for _, driver in features_df.iterrows():
            tier = driver['discount_tier']
            discount_rate = discount_rates[tier]
            final_premium = base_premium * (1 - discount_rate)
            annual_savings = base_premium - final_premium

            premiums.append(final_premium)
            savings_list.append(annual_savings)

        features_df['estimated_premium'] = premiums
        features_df['annual_savings'] = savings_list

        return features_df

    def add_realistic_noise(self, X):
        """Add measurement noise and missing values (moderate noise for larger dataset)"""
        X_noisy = X.copy()

        for col in X_noisy.columns:
            if col in ['driver_id', 'driver_profile', 'discount_eligible', 'discount_tier']:
                continue

            if 'speed' in col.lower() and 'score' not in col.lower():
                noise = np.random.normal(0, X_noisy[col].std() * 0.08, len(X_noisy))
                X_noisy[col] = X_noisy[col] + noise
            elif 'acceleration' in col.lower() or 'braking' in col.lower():
                noise = np.random.normal(0, X_noisy[col].std() * 0.12, len(X_noisy))
                X_noisy[col] = X_noisy[col] + noise
            elif 'ratio' in col.lower() or 'events' in col.lower():
                noise = np.random.normal(0, X_noisy[col].std() * 0.06, len(X_noisy))
                X_noisy[col] = X_noisy[col] + noise


        numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mask = np.random.random(len(X_noisy)) < 0.03
            X_noisy.loc[mask, col] = np.nan


        X_noisy = X_noisy.fillna(X_noisy.median())


        speed_cols = [col for col in X_noisy.columns if 'speed' in col.lower() and 'score' not in col.lower()]
        for col in speed_cols:
            X_noisy[col] = np.maximum(X_noisy[col], 0)

        score_cols = [col for col in X_noisy.columns if 'score' in col.lower()]
        for col in score_cols:
            X_noisy[col] = np.clip(X_noisy[col], 0, 100)

        return X_noisy

    def create_edge_cases(self, df):
        """Create borderline cases for discount eligibility (reduced ratio for larger dataset)"""
        edge_cases = []

        num_edge_cases = len(df) // 10

        for i in range(num_edge_cases):
            base_idx = np.random.randint(len(df))
            case = df.iloc[base_idx].copy()


            case_type = np.random.choice(['borderline_qualify', 'borderline_miss', 'mixed_signals'])

            if case_type == 'borderline_qualify':

                case['overall_safety_score'] = np.random.uniform(74, 78)
                case['avg_speed_kph'] = np.random.uniform(68, 72)
                case['harsh_braking_events'] = np.random.randint(4, 6)
                case['harsh_acceleration_events'] = np.random.randint(3, 5)
                case['phone_usage_events'] = np.random.randint(0, 2)
                case['discount_eligible'] = np.random.choice([True, False], p=[0.6, 0.4])
                case['discount_tier'] = 1 if case['discount_eligible'] else 0

            elif case_type == 'borderline_miss':

                case['overall_safety_score'] = np.random.uniform(70, 74)
                case['avg_speed_kph'] = np.random.uniform(72, 76)
                case['harsh_braking_events'] = np.random.randint(6, 9)
                case['harsh_acceleration_events'] = np.random.randint(5, 8)
                case['phone_usage_events'] = np.random.randint(1, 3)
                case['discount_eligible'] = np.random.choice([True, False], p=[0.3, 0.7])
                case['discount_tier'] = 1 if case['discount_eligible'] else 0

            else:

                case['overall_safety_score'] = np.random.uniform(72, 76)
                case['avg_speed_kph'] = np.random.uniform(58, 62)
                case['harsh_braking_events'] = np.random.randint(7, 10)
                case['phone_usage_events'] = np.random.randint(0, 1)
                case['night_driving_ratio'] = np.random.uniform(0.2, 0.4)
                case['discount_eligible'] = np.random.choice([True, False])
                case['discount_tier'] = np.random.choice([0, 1])

            case['driver_id'] = f"EDGE_{i:04d}"
            case['driver_profile'] = 'borderline'
            edge_cases.append(case)

        return pd.DataFrame(edge_cases)

    def apply_smote_enn_resampling(self, X, y):
        """Apply SMOTE-ENN for class balance (conservative for larger dataset)"""
        print(f"Original discount distribution:")
        print(f"  Discount Eligible: {sum(y)} drivers ({sum(y) / len(y) * 100:.1f}%)")
        print(f"  Standard Rate: {len(y) - sum(y)} drivers ({(len(y) - sum(y)) / len(y) * 100:.1f}%)")


        smote_enn = SMOTEENN(
            smote=SMOTE(
                sampling_strategy=0.6,
                k_neighbors=5,
                random_state=42
            ),
            enn=EditedNearestNeighbours(
                n_neighbors=3,
                kind_sel='mode'
            ),
            random_state=42
        )


        X_resampled, y_resampled = smote_enn.fit_resample(X, y)

        print(f"After SMOTE-ENN:")
        print(f"  Discount Eligible: {sum(y_resampled)} drivers ({sum(y_resampled) / len(y_resampled) * 100:.1f}%)")
        print(
            f"  Standard Rate: {len(y_resampled) - sum(y_resampled)} drivers ({(len(y_resampled) - sum(y_resampled)) / len(y_resampled) * 100:.1f}%)")
        print(f"Data size: {len(X)} → {len(X_resampled)}")

        return X_resampled, y_resampled

    def process_complete_pipeline(self, raw_data_path="data/telematics_data.csv"):
        """Complete feature engineering pipeline"""


        if not os.path.exists(raw_data_path):
            print(f"Raw data not found at {raw_data_path}")
            print("   Run 'python data_simulator.py' first!")
            return None, None

        raw_df = pd.read_csv(raw_data_path)


        features_df = self.create_features_from_raw_data(raw_df)


        features_df = self.determine_discount_eligibility(features_df)
        features_df = self.calculate_premium_examples(features_df)


        edge_cases_df = self.create_edge_cases(features_df)
        combined_df = pd.concat([features_df, edge_cases_df], ignore_index=True)


        feature_cols = [col for col in combined_df.columns
                        if col not in ['driver_id', 'driver_profile', 'discount_eligible',
                                       'discount_tier', 'estimated_premium', 'annual_savings']]
        X = combined_df[feature_cols]
        y = combined_df['discount_eligible'].astype(int).values


        X_balanced, y_balanced = self.apply_smote_enn_resampling(X, y)


        X_final = self.add_realistic_noise(pd.DataFrame(X_balanced, columns=X.columns))


        ml_output_path = "data/ml_ready_dataset.csv"
        business_output_path = "data/business_analysis.csv"

        ml_df = X_final.copy()
        ml_df['discount_eligible'] = y_balanced
        ml_df.to_csv(ml_output_path, index=False)


        combined_df.to_csv(business_output_path, index=False)


        return X_final, y_balanced


if __name__ == "__main__":
    print(" Starting Telematics Feature Engineering")

    engineer = TelematicsFeatureEngineer()
    X_ml, y_ml = engineer.process_complete_pipeline()

    if X_ml is not None:
        print("\n Feature Engineering Summary:")
        print(f"   Total ML samples: {len(X_ml)}")
        print(f"   Discount eligible: {sum(y_ml)} ({sum(y_ml) / len(y_ml) * 100:.1f}%)")
        print(f"   Standard rate: {len(y_ml) - sum(y_ml)} ({(len(y_ml) - sum(y_ml)) / len(y_ml) * 100:.1f}%)")
        print("Command: python src/train_model.py")