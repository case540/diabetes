import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
from torch import nn

class CGMDataset(Dataset):
    """
    PyTorch Dataset for CGM data.
    Handles data loading, preprocessing, and windowing.
    """
    def __init__(self, df, scaler, patient_ids, window_size_hours=48, slide_hours=6, is_train=False):
        """
        Args:
            df (pd.DataFrame): The full dataframe.
            scaler (StandardScaler): The scaler fitted on the training data.
            patient_ids (list): List of patient IDs to include in this dataset.
            window_size_hours (int): The size of the window in hours.
            slide_hours (int): The number of hours to slide the window.
            is_train (bool): If True, apply data augmentation.
        """
        self.window_size_hours = window_size_hours
        self.slide_hours = slide_hours
        
        # Calculate samples per window
        self.samples_per_window = int(self.window_size_hours * 60 / 5) # 5-minute intervals
        self.slide_samples = int(self.slide_hours * 60 / 5)

        self.scaler = scaler
        self.is_train = is_train

        self.sequences = []
        self.labels = []

        # Filter the dataframe for the specified patients
        df_subset = df[df['ID'].isin(patient_ids)].copy()

        # Apply scaling and create time-of-day feature
        df_subset['reading_scaled'] = self.scaler.transform(df_subset[['reading']])
        df_subset['time_of_day'] = (df_subset['time'].dt.hour * 60 + df_subset['time'].dt.minute) / (24 * 60)

        # Process each patient
        for patient_id, group in df_subset.groupby('ID'):
            # Get the single, consistent label for the patient BEFORE resampling
            # to prevent it from being corrupted by interpolation.
            patient_label = group['label'].iloc[0]

            # Set time as index for resampling
            group = group.set_index('time')

            # Resample and interpolate ONLY the glucose reading. This is more robust.
            readings_resampled = group[['reading']].resample('5T').interpolate(method='linear')

            # Create the new, clean group dataframe from the resampled readings
            group = readings_resampled.reset_index()

            # Re-calculate features for the new resampled data
            # Important: Use the same scaler to transform the resampled data
            group['reading_scaled'] = self.scaler.transform(group[['reading']].fillna(group[['reading']].mean()))
            group['time_of_day'] = (group['time'].dt.hour * 60 + group['time'].dt.minute) / (24 * 60)
            
            # Extract features for the patient
            features = group[['reading_scaled', 'time_of_day']].values
            
            # Create sliding windows
            for i in range(0, len(features) - self.samples_per_window + 1, self.slide_samples):
                sequence = features[i:i+self.samples_per_window]
                self.sequences.append(sequence)
                self.labels.append(patient_label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Make a copy to prevent modifying the original data in memory
        sequence = self.sequences[idx].copy()
        label = self.labels[idx]

        if self.is_train:
            seq_len = sequence.shape[0]
            
            # 1. Dropout (1% of sequence)
            # We set the scaled reading to 0, which is the mean of the training data.
            dropout_indices = np.random.choice(seq_len, size=int(seq_len * 0.01), replace=False)
            sequence[dropout_indices, 0] = 0

            # 2. Perturbation (2% of sequence)
            perturb_indices = np.random.choice(seq_len, size=int(seq_len * 0.02), replace=False)
            
            # Add small noise to glucose reading
            glucose_noise = np.random.normal(0, 0.1, size=len(perturb_indices))
            sequence[perturb_indices, 0] += glucose_noise
            
            # Add small noise to time of day (up to ~5 minutes)
            # 5 minutes / (24 * 60 minutes) = 0.00347
            time_noise = np.random.uniform(-0.0035, 0.0035, size=len(perturb_indices))
            sequence[perturb_indices, 1] += time_noise
            
            # Ensure time_of_day feature stays within [0, 1]
            sequence[:, 1] = np.clip(sequence[:, 1], 0, 1)

        return torch.FloatTensor(sequence), torch.tensor(label, dtype=torch.long)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class PrediabetesTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super(PrediabetesTransformer, self).__init__()
        self.d_model = d_model
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Input embedding layer
        self.input_embed = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Classifier Head
        self.classifier = nn.Linear(d_model, 2) # 2 classes: 0 and 1

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        
        # Prepend CLS token
        batch_size = src.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Embed input
        src = self.input_embed(src)
        
        # Combine CLS token and embedded source
        src_with_cls = torch.cat([cls_tokens, src], dim=1)
        
        # Add positional encoding
        src_with_cls = self.pos_encoder(src_with_cls.transpose(0, 1)).transpose(0, 1)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src_with_cls)
        
        # Get the CLS token output (first token)
        cls_output = output[:, 0, :]
        
        # Pass through classifier
        logits = self.classifier(cls_output)
        
        return logits

# Example usage:
if __name__ == "__main__":
    # --- This example usage is now illustrative ---
    # In practice, the train script will handle the splitting and dataset creation.
    print("--- Testing CGMDataset ---")
    df_raw = pd.read_csv('cgm_data.csv')

    # Standardize the dataframe
    df = pd.DataFrame()
    df['time'] = pd.to_datetime(df_raw['Time'])
    df['ID'] = df_raw['Subject']
    df['reading'] = df_raw['Gl']
    df['label'] = df_raw['Label'].map({'pre': 1, 'non': 0})

    patient_ids = df['ID'].unique()
    train_ids = patient_ids[:15] # Dummy split

    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    scaler.fit(df[df['ID'].isin(train_ids)][['reading']])

    # Create the dataset
    dataset = CGMDataset(df=df, scaler=scaler, patient_ids=train_ids)
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get one batch
    seq_batch, label_batch = next(iter(dataloader))
    
    print("Successfully created DataLoader.")
    print(f"Number of windows (samples): {len(dataset)}")
    print(f"Batch of sequences shape: {seq_batch.shape}")
    print(f"Batch of labels shape: {label_batch.shape}")

    # --- Model Test ---
    print("\n--- Testing Model ---")
    model = PrediabetesTransformer()
    
    # Pass batch through model
    output_logits = model(seq_batch)
    
    print("Model ran successfully.")
    print(f"Output logits shape: {output_logits.shape}")

    # Expected output shape: [batch_size, num_classes]
    expected_output_shape = (4, 2)
    assert output_logits.shape == expected_output_shape, f"Shape mismatch: {output_logits.shape} != {expected_output_shape}"
    print("Output shape is correct.") 