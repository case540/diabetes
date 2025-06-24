import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CGMformerFinetuneDataset(Dataset):
    """
    Dataset class to prepare CGM data for fine-tuning the CGMformer model.
    This class handles:
    1. Loading continuous glucose data.
    2. Rounding glucose values to the nearest integer.
    3. Tokenizing integers using the CGMformer's vocabulary.
    4. Creating fixed-size windows (288 samples for 24 hours).
    """
    def __init__(self, cgm_df, bio_df, token2id_path, use_interpolation=True, window_samples=288, slide_samples=288, patient_ids=None):
        self.window_samples = window_samples
        self.slide_samples = slide_samples
        self.sequences = []
        self.labels = []

        # --- 1. Load Tokenizer ---
        logging.info(f"Loading tokenizer from: {token2id_path}")
        try:
            with open(token2id_path, 'rb') as f:
                self.token2id = pickle.load(f)
            self.pad_token_id = self.token2id.get('<pad>', self.token2id.get('<PAD>', 0))
        except FileNotFoundError:
            logging.error(f"Tokenizer file not found at {token2id_path}. Cannot proceed.")
            raise

        # --- 2. Merge dataframes to get labels for each CGM reading ---
        # We need to map health_condition from bio_df to each patient's CGM data
        labeled_cgm_data = pd.merge(cgm_df, bio_df[['subject', 'health_condition']], on='subject', how='left')
        
        # --- Filter for specific patients if provided ---
        if patient_ids:
            logging.info(f"Dataset processing {len(patient_ids)} specific patients.")
            labeled_cgm_data = labeled_cgm_data[labeled_cgm_data['subject'].isin(patient_ids)]
        
        # --- 3. Create a mapping for labels to integers ---
        self.label_map = {label: i for i, label in enumerate(sorted(labeled_cgm_data['health_condition'].unique()))}
        logging.info(f"Label mapping: {self.label_map}")

        # --- 4. Process each patient's data ---
        for subject_id, group in labeled_cgm_data.groupby('subject'):
            # Convert time column to datetime if it's not already
            group['time'] = pd.to_datetime(group['time'])
            
            if use_interpolation:
                logging.info(f"Resampling data for subject {subject_id} to 5-minute intervals.")
                # --- Robust Resampling Step ---
                # We separate numeric interpolation from categorical data handling to avoid errors.
                group_indexed = group.set_index('time')

                # 1. Resample and interpolate ONLY the numeric 'gl' column.
                #    .mean() aggregates any existing data points within a 5-min window.
                gl_resampled = group_indexed['gl'].resample('5T').mean().interpolate(method='linear')

                # 2. Create a new dataframe from the clean, resampled glucose series.
                group = pd.DataFrame(gl_resampled)
                
                # 3. Get the single, constant label for the patient and assign it to all rows.
                patient_label = group_indexed['health_condition'].iloc[0]
                group['health_condition'] = patient_label

                group = group.reset_index()
            else:
                logging.info(f"Using simplified processing for subject {subject_id} (no interpolation).")
                # Simplified mode: just sort by time and assume intervals are correct
                group = group.sort_values(by='time').reset_index(drop=True)

            # --- Tokenization Step ---
            # Round glucose to nearest int, handle missing values, and convert to token ID
            glucose_values = group['gl'].round().astype('Int64') # Use Int64 to handle potential NaNs
            
            # Convert glucose integers to token IDs, using pad_token for unknown values
            token_ids = [self.token2id.get(val, self.pad_token_id) for val in glucose_values]

            # --- Windowing Step ---
            for i in range(0, len(token_ids) - self.window_samples + 1, self.slide_samples):
                sequence = token_ids[i : i + self.window_samples]
                
                # Get the single label for this patient
                label = self.label_map[group['health_condition'].iloc[0]]
                
                self.sequences.append(torch.tensor(sequence, dtype=torch.long))
                self.labels.append(torch.tensor(label, dtype=torch.long))

        logging.info(f"Created {len(self.sequences)} sequences from the data.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

if __name__ == '__main__':
    # This is a placeholder for a test case to show how to use the dataset
    print("This script defines the Dataset for fine-tuning. It should be imported by the main training script.")
    print("Example usage would involve:")
    print("1. Loading the CGM and bio data from the new_workflow analysis script.")
    print("2. Instantiating this dataset: `dataset = CGMformerFinetuneDataset(all_cgm_data, bio_df, 'cgmformer_ckpt/cgm_ckp/token2id.pkl')`")
    print("3. Passing it to a PyTorch DataLoader.") 