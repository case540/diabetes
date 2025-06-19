import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_patient_data(csv_file, patient_id):
    """
    Loads CGM data and plots the glucose readings over time for a specific patient.

    Args:
        csv_file (str): Path to the CGM data CSV file.
        patient_id (str): The ID of the patient to plot.
    """
    # Set plot style
    sns.set_theme(style="darkgrid")

    # Load data
    try:
        df_raw = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.")
        return

    # Standardize the dataframe
    df = pd.DataFrame()
    df['time'] = pd.to_datetime(df_raw['Time'])
    df['ID'] = df_raw['Subject']
    df['reading'] = df_raw['Gl']
    df['label'] = df_raw['Label'].map({'pre': 1, 'non': 0})

    # Filter for the specific patient
    patient_df = df[df['ID'] == patient_id]

    if patient_df.empty:
        print(f"Error: No data found for patient ID '{patient_id}'.")
        print(f"Available patient IDs: {df['ID'].unique()}")
        return

    # Get label for the title
    label = "Prediabetic" if patient_df['label'].iloc[0] == 1 else "Not Prediabetic"
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=patient_df, x='time', y='reading', marker='o', markersize=4)
    
    # Formatting
    plt.title(f'Glucose Readings for {patient_id} ({label})')
    plt.xlabel('Time')
    plt.ylabel('Glucose Reading (mg/dL)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    # This is an example of how to use the function.
    # It will plot the data for 'patient_1'.
    # You can change this to any patient ID from your generated data.
    DATA_FILE = 'cgm_data.csv'
    
    # First, let's find out which patients are available
    try:
        all_ids = pd.read_csv(DATA_FILE)['Subject'].unique()
        print(f"Available patient IDs: {all_ids}")
        
        if len(all_ids) > 0:
            # Plot data for the first available patient
            plot_patient_data(DATA_FILE, all_ids[0])
    except FileNotFoundError:
        print(f"Could not find {DATA_FILE}. Please run src/data_generation.py first.") 