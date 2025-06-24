import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os
import logging
from .tokenizer import CGMTokenizer

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
    def __init__(self, cgm_df, bio_df, token2id_path, use_interpolation=True, patient_ids=None, is_train=False, aug_prob=0.5, jitter_strength=3.0):
        self.tokenizer = CGMTokenizer(token2id_path=token2id_path)
        self.sequences = []
        self.labels = []
        self.is_train = is_train
        self.aug_prob = aug_prob
        self.jitter_strength = jitter_strength

        # --- Data Preprocessing ---
        # Get the label for each subject
        bio_df = bio_df.set_index('subject')['health_condition'].to_dict()
        
        # This mapping should be consistent across train and val datasets
        self.label_map = {label: i for i, label in enumerate(sorted(np.unique(list(bio_df.values()))))}
        logging.info(f"Label mapping: {self.label_map}")

        # If specific patient_ids are provided, filter the data
        if patient_ids:
            labeled_cgm_data = cgm_df[cgm_df['subject'].isin(patient_ids)].copy()
            logging.info(f"Dataset processing {len(patient_ids)} specific patients.")
        else:
            labeled_cgm_data = cgm_df.copy()

        labeled_cgm_data['health_condition'] = labeled_cgm_data['subject'].map(bio_df)

        # Process each patient's data
        for subject_id, group in labeled_cgm_data.groupby('subject'):
            # Convert time column to datetime if it's not already
            group['time'] = pd.to_datetime(group['time'])
            
            if use_interpolation:
                logging.info(f"Resampling data for subject {subject_id} to 5-minute intervals.")
                # --- Robust Resampling Step ---
                group_indexed = group.set_index('time')
                gl_resampled = group_indexed['gl'].resample('5T').mean().interpolate(method='linear')
                group = pd.DataFrame(gl_resampled)
                patient_label = group_indexed['health_condition'].iloc[0]
                group['health_condition'] = patient_label
                group = group.reset_index()
            else:
                logging.info(f"Using simplified processing for subject {subject_id} (no interpolation).")
                group = group.sort_values(by='time')

            # Process each 24-hour window
            for i in range(0, len(group) - self.tokenizer.seq_len + 1, self.tokenizer.seq_len):
                sequence_df = group.iloc[i:i + self.tokenizer.seq_len]
                
                if len(sequence_df) == self.tokenizer.seq_len:
                    self.sequences.append(sequence_df['gl'].values)
                    label = self.label_map[sequence_df['health_condition'].iloc[0]]
                    self.labels.append(label)

        logging.info(f"Created {len(self.sequences)} sequences from the data.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].copy() # Use a copy to avoid modifying the original data

        # --- On-the-fly Data Augmentation (Jitter) ---
        if self.is_train and np.random.rand() < self.aug_prob:
            # Add small random noise to a subset of the glucose readings
            jitter = np.random.uniform(-self.jitter_strength, self.jitter_strength, size=sequence.shape)
            # Apply jitter to ~15% of the points
            mask = np.random.rand(*sequence.shape) < 0.15
            sequence = sequence + (jitter * mask)

        # Tokenize the glucose sequence
        tokenized_sequence = self.tokenizer.tokenize(list(sequence))
        
        # Convert to tensor
        return torch.tensor(tokenized_sequence, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

if __name__ == '__main__':
    # This is a placeholder for a test case to show how to use the dataset
    print("This script defines the Dataset for fine-tuning. It should be imported by the main training script.")
    print("Example usage would involve:")
    print("1. Loading the CGM and bio data from the new_workflow analysis script.")
    print("2. Instantiating this dataset: `dataset = CGMformerFinetuneDataset(all_cgm_data, bio_df, 'cgmformer_ckpt/cgm_ckp/token2id.pkl')`")
    print("3. Passing it to a PyTorch DataLoader.") 