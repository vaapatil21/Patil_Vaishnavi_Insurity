""" Code summary - This script is a data simulator that programmatically generates a
high-fidelity, synthetic telematics dataset for model development.
It creates realistic driving data for 800 drivers across four distinct risk profiles
to serve as a substitute for real-world data. The final output is a single CSV file
containing detailed trip information, which acts as the foundational input for the feature engineering pipeline.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random



def create_trip_timestamps(duration_minutes):
    """Creates a series of timestamps for a trip."""
    num_points = duration_minutes * 6
    start_time = datetime.now() - timedelta(days=random.randint(0, 90),
                                            hours=random.randint(0, 23))
    return pd.date_range(start=start_time, periods=num_points, freq='10S')


def simulate_gps_movement(num_points, speed_kph):
    """Simulates realistic GPS lat/lon changes based on speed."""
    lat, lon = 43.1566, -77.6088
    lat_deltas = np.random.normal(0, 0.00001 * speed_kph, num_points).cumsum()
    lon_deltas = np.random.normal(0, 0.00001 * speed_kph, num_points).cumsum()

    return lat + lat_deltas, lon + lon_deltas


def generate_driver_trips(driver_id, profile, num_trips):
    """Generates all trips for a single driver based on their profile."""
    driver_profiles = {
        'excellent': {'avg_speed_kph': 55, 'speed_var': 8, 'harsh_event_prob': 0.01},
        'good': {'avg_speed_kph': 65, 'speed_var': 12, 'harsh_event_prob': 0.03},
        'average': {'avg_speed_kph': 75, 'speed_var': 15, 'harsh_event_prob': 0.08},
        'poor': {'avg_speed_kph': 85, 'speed_var': 20, 'harsh_event_prob': 0.15}
    }

    driver_spec = driver_profiles[profile]
    trips_data = []

    for i in range(num_trips):
        duration = random.randint(10, 75)
        timestamps = create_trip_timestamps(duration)
        num_points = len(timestamps)

        speed = np.random.normal(driver_spec['avg_speed_kph'], driver_spec['speed_var'], num_points)
        speed = np.clip(speed, 0, 160)


        speed_ms = speed * (1000 / 3600)
        accel = np.diff(speed_ms, prepend=speed_ms[0]) / 10

        harsh_braking = (accel < -2.5) & (np.random.rand(num_points) < driver_spec['harsh_event_prob'])
        harsh_accel = (accel > 2.5) & (np.random.rand(num_points) < driver_spec['harsh_event_prob'])

        lats, lons = simulate_gps_movement(num_points, speed)

        trip_df = pd.DataFrame({
            'timestamp': timestamps,
            'latitude': lats,
            'longitude': lons,
            'speed_kph': speed,
            'acceleration_ms2': accel,
            'harsh_braking': harsh_braking,
            'harsh_acceleration': harsh_accel
        })

        trip_df['driver_id'] = driver_id
        trip_df['driver_profile'] = profile
        trip_df['trip_id'] = f"{driver_id}_T{i:03d}"
        trips_data.append(trip_df)

    return pd.concat(trips_data, ignore_index=True)


if __name__ == "__main__":
    print("Generating synthetic telematics data")

    NUM_DRIVERS = 800
    TRIPS_PER_DRIVER = 15

    all_drivers_data = []

    for i in range(NUM_DRIVERS):
        driver_id = f"D{i:04d}"

        profile = np.random.choice(
            ['excellent', 'good', 'average', 'poor'],
            p=[0.35, 0.30, 0.25, 0.10]
        )

        driver_df = generate_driver_trips(driver_id, profile, TRIPS_PER_DRIVER)
        all_drivers_data.append(driver_df)

        if (i + 1) % 100 == 0:
            print(f"   Generated {i + 1}/{NUM_DRIVERS} drivers...")

    final_df = pd.concat(all_drivers_data, ignore_index=True)

    output_path = "data/telematics_data.csv"
    os.makedirs('data', exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"   Total Drivers: {final_df['driver_id'].nunique()}")
    print(f"   Total Trips: {final_df['trip_id'].nunique()}")
    print(f"   Total Data Points: {len(final_df)}")
    print("Command: python src/feature_engineering.py")