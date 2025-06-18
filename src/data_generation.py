import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_cgm_data(n_patients=10, n_days=14):
    """
    Generates synthetic CGM data for a number of patients over a number of days.

    Args:
        n_patients (int): The number of patients.
        n_days (int): The number of days to generate data for.

    Returns:
        pandas.DataFrame: A dataframe with columns ['Subject', 'Date', 'Time', 'Gl', 'Label']
    """
    data = []
    start_date = datetime.now()

    for i in range(n_patients):
        patient_id = f"patient_{i+1}"
        # Randomly assign a label
        is_prediabetic = np.random.choice([0, 1])

        # Set glucose profile parameters based on label
        if is_prediabetic:
            # Higher baseline and more variability for prediabetic patients
            baseline_glucose = np.random.normal(110, 10)
            meal_spike_scale = np.random.uniform(30, 60)
            spike_duration_scale = np.random.uniform(2, 4)
        else:
            # Normal glucose profile
            baseline_glucose = np.random.normal(85, 5)
            meal_spike_scale = np.random.uniform(15, 30)
            spike_duration_scale = np.random.uniform(1, 2)

        for day in range(n_days):
            for minute in range(0, 24 * 60, 5):
                current_time = start_date + timedelta(days=day, minutes=minute)
                
                # Simulate meal spikes (e.g., at 8am, 1pm, 7pm)
                reading = baseline_glucose
                hour = current_time.hour
                # Simple meal simulation
                if 8 <= hour < 10 or 13 <= hour < 15 or 19 <= hour < 21:
                    time_since_meal = (hour % 12 * 60 + current_time.minute) - (8*60 if 8 <= hour < 10 else 13*60 if 13 <= hour < 15 else 19*60)
                    spike = meal_spike_scale * np.exp(-((time_since_meal / (spike_duration_scale * 60))**2))
                    reading += spike

                # Add some noise
                reading += np.random.normal(0, 2)

                data.append({
                    "Subject": patient_id,
                    "Date": current_time.strftime('%m/%d/%Y'),
                    "Time": current_time.strftime('%H:%M:%S'),
                    "Gl": reading,
                    "Label": "pre" if is_prediabetic else "non"
                })

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_cgm_data(n_patients=20, n_days=14)
    print(df.head())
    print(f"\nGenerated {len(df)} data points for {df['Subject'].nunique()} patients.")
    
    # Save to a csv file
    df.to_csv("cgm_data.csv", index=False)
    print("Data saved to cgm_data.csv") 