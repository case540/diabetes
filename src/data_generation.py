import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_cgm_data(n_patients=10, n_days=14):
    """
    Generates synthetic CGM data for a number of patients over a number of days.

    Args:
        n_patients (int): The number of patients.
        n_days (int): The number of days to generate data for.

    Returns:
        pandas.DataFrame: A dataframe with columns ['Subject', 'Time', 'Gl', 'Label']
    """
    data = []
    start_date = datetime(2025, 6, 19, 10, 35, 43)

    for i in range(1, n_patients + 1):
        patient_id = f"patient_{i}"
        label = "pre" if i % 2 == 1 else "non"
        base_glucose = 120 if label == "pre" else 90
        
        # --- Introduce a random time gap for some patients ---
        gap_start_minute = -1
        if np.random.rand() < 0.5: # 50% chance of having a gap
            gap_duration_hours = np.random.randint(2, 8) # 2-7 hour gap
            gap_duration_minutes = gap_duration_hours * 60
            
            # Find a random start time for the gap, ensuring it doesn't go past the end
            max_start_minute = (n_days * 24 * 60) - gap_duration_minutes
            gap_start_minute = np.random.randint(0, max_start_minute)
            gap_end_minute = gap_start_minute + gap_duration_minutes
            print(f"INFO: Patient {i} will have a {gap_duration_hours}hr gap starting at minute {gap_start_minute}.")

        for day in range(n_days):
            for minute in range(0, 24 * 60, 5):
                total_minutes_elapsed = day * 24 * 60 + minute
                
                # Check if we are inside the gap
                if gap_start_minute != -1 and gap_start_minute <= total_minutes_elapsed < gap_end_minute:
                    continue

                # Simulate device connectivity issues by randomly dropping some points
                if np.random.rand() < 0.05: # 5% chance to skip a reading
                    continue

                current_time = start_date + timedelta(days=day, minutes=minute)
                
                # Simulate meal spikes (e.g., at 8am, 1pm, 7pm)
                reading = base_glucose
                hour = current_time.hour
                # Simple meal simulation
                if 8 <= hour < 10 or 13 <= hour < 15 or 19 <= hour < 21:
                    time_since_meal = (hour % 12 * 60 + current_time.minute) - (8*60 if 8 <= hour < 10 else 13*60 if 13 <= hour < 15 else 19*60)
                    meal_spike_scale = random.uniform(20, 40)
                    spike_duration_scale = random.uniform(0.5, 1.5)
                    spike = meal_spike_scale * np.exp(-((time_since_meal / (spike_duration_scale * 60))**2))
                    reading += spike

                # Add some noise
                reading += np.random.normal(0, 2)

                data.append({
                    "Subject": patient_id,
                    "Time": current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "Gl": reading,
                    "Label": label
                })

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_cgm_data(n_patients=20, n_days=14)
    print(df.head())
    print(f"\nGenerated {len(df)} data points for {df['Subject'].nunique()} patients.")
    
    # Save to a csv file
    df.to_csv("cgm_data.csv", index=False)
    print("Data saved to cgm_data.csv") 