import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from src.model import CGMDataset, PrediabetesTransformer
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
import os
from src.data_generation import generate_cgm_data
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix

def create_collate_fn(pad_value=0):
    """
    Creates a collate function for the DataLoader to handle variable-length sequences.
    """
    def collate_fn(batch):
        # Separate sequences and labels
        sequences, labels = zip(*batch)
        
        # Get sequence lengths
        lengths = torch.tensor([len(seq) for seq in sequences])
        
        # Pad sequences
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
        
        # Create attention mask (True for padding, False for real data)
        # It's the opposite of what PyTorch's Transformer expects, but we'll flip it later.
        mask = (padded_sequences[:, :, 0] == pad_value)
        
        # Stack labels
        labels = torch.stack(labels)
        
        return padded_sequences, labels, mask

    return collate_fn

# --- Configuration ---
def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer model for Prediabetes Prediction.")
    parser.add_argument('--data_file', type=str, default='cgm_data.csv', help='Path to the CGM data file.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Proportion of data to use for validation.')
    parser.add_argument('--run_name', type=str, default='experiment_1', help='Name for the TensorBoard run.')
    return parser.parse_args()

def train_model(args):
    """
    Main function to train the model.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- TensorBoard Initialization ---
    writer = SummaryWriter(f'runs/{args.run_name}')

    # --- Data Loading and Patient-Aware Splitting ---

    # If the default data file is specified but doesn't exist, generate it.
    if args.data_file == 'cgm_data.csv' and not os.path.exists(args.data_file):
        print(f"Default data file '{args.data_file}' not found. Generating it now...")
        df_generated = generate_cgm_data()
        df_generated.to_csv(args.data_file, index=False)
        print(f"'{args.data_file}' created successfully.\n")

    try:
        df_raw = pd.read_csv(args.data_file)
    except FileNotFoundError:
        print(f"Error: The data file '{args.data_file}' was not found.")
        print("Please check the file path or run 'python -m src.data_generation' to create the default sample data.")
        return
    
    # --- Handle duplicate timestamps before standardization ---
    # A patient may have multiple readings at the exact same time. We average them.
    initial_rows = len(df_raw)
    # The 'first' aggregation for Label and Date assumes they are consistent for a given Subject.
    # We reset the index to bring 'Subject' and 'Time' back as columns.
    df_raw = df_raw.groupby(['Subject', 'Time']).agg({
        'Gl': 'mean',
        'Label': 'first'
    }).reset_index()
    final_rows = len(df_raw)

    if initial_rows > final_rows:
        print(f"\nINFO: Found and averaged {initial_rows - final_rows} duplicate timestamp entries in the data.")

    # Standardize the dataframe
    df = pd.DataFrame()
    df['time'] = pd.to_datetime(df_raw['Time'])
    df['ID'] = df_raw['Subject']
    df['reading'] = df_raw['Gl']
    df['label'] = df_raw['Label'].map({'pre': 1, 'non': 0})
    
    # --- Stratified, Patient-Aware Splitting ---
    # Get unique patient IDs and their corresponding labels
    patient_label_map = df[['ID', 'label']].drop_duplicates()

    # --- Data Integrity Check ---
    # A patient should not have multiple, conflicting labels.
    patient_id_counts = patient_label_map['ID'].value_counts()
    inconsistent_patients = patient_id_counts[patient_id_counts > 1].index.tolist()

    if inconsistent_patients:
        print("\n!!! DATA INTEGRITY ERROR !!!")
        print("The following patients have been assigned multiple, conflicting labels (e.g., 'pre' and 'non'):")
        for patient in inconsistent_patients:
            print(f"  - {patient}")
        print("\nPlease clean the data to ensure each patient has only one label, then try again.")
        return # Exit the training script

    patient_labels = patient_label_map.set_index('ID')
    
    # Separate patients by label
    pre_patients = patient_labels[patient_labels['label'] == 1].index.tolist()
    non_patients = patient_labels[patient_labels['label'] == 0].index.tolist()

    # Shuffle each list independently
    np.random.shuffle(pre_patients)
    np.random.shuffle(non_patients)

    # Calculate split point for each class
    pre_split_idx = int(len(pre_patients) * (1 - args.val_split))
    non_split_idx = int(len(non_patients) * (1 - args.val_split))

    # Create train/val splits for each class
    train_pre = pre_patients[:pre_split_idx]
    val_pre = pre_patients[pre_split_idx:]
    train_non = non_patients[:non_split_idx]
    val_non = non_patients[non_split_idx:]

    # Combine and shuffle the final patient lists
    train_patient_ids = np.array(train_pre + train_non)
    val_patient_ids = np.array(val_pre + val_non)
    np.random.shuffle(train_patient_ids)
    np.random.shuffle(val_patient_ids)

    # Fit scaler ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    train_readings = df[df['ID'].isin(train_patient_ids)][['reading']]
    scaler.fit(train_readings)

    # Create datasets
    train_dataset = CGMDataset(df=df, scaler=scaler, patient_ids=train_patient_ids, is_train=True)
    val_dataset = CGMDataset(df=df, scaler=scaler, patient_ids=val_patient_ids, is_train=False)

    collate_fn = create_collate_fn()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Data loaded. Train patients: {len(train_patient_ids)}, Validation patients: {len(val_patient_ids)}")
    print(f"Train samples (windows): {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- Model Initialization ---
    model = PrediabetesTransformer().to(DEVICE)
    print("\n--- Model Architecture ---")
    print(model)
    print("--------------------------\n")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (sequences, labels, masks) in enumerate(train_loader):
            sequences, labels, masks = sequences.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(sequences, src_key_padding_mask=masks)
            loss = criterion(outputs, labels)

            # Check for NaN loss
            if torch.isnan(loss):
                print("\n!!! NaN loss detected! !!!")
                print("This can be caused by very high learning rates or numerical instability.")
                print("Skipping this batch. Consider lowering the learning rate.")
                continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # Log every 10 mini-batches
                global_step = epoch * len(train_loader) + i
                avg_loss_so_far = running_loss / 10
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss_so_far:.4f}')
                writer.add_scalar('Loss/train', avg_loss_so_far, global_step)
                running_loss = 0.0
        
        # --- Validation Step ---
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, DEVICE, writer, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    print("\n--- Finished Training ---")
    writer.close()
    
    # Save the trained model (optional)
    # torch.save(model.state_dict(), f'{args.run_name}_model.pth')
    # print(f"Model saved to {args.run_name}_model.pth")


def evaluate_model(model, dataloader, criterion, device, writer, epoch):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for sequences, labels, masks in dataloader:
            sequences, labels, masks = sequences.to(device), labels.to(device), masks.to(device)
            
            outputs = model(sequences, src_key_padding_mask=masks)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_samples

    print(f'\nValidation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Calculate FP/FN from confusion matrix, assuming class 1 is the "positive" class (prediabetic)
    try:
        # tn, fp, fn, tp
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        print(f'           - False Positives (FP): {fp} (Predicted "pre", but was "non")')
        print(f'           - False Negatives (FN): {fn} (Predicted "non", but was "pre")\n')
        
        # Log FP and FN to TensorBoard
        writer.add_scalar('Validation/False_Positives', fp, epoch)
        writer.add_scalar('Validation/False_Negatives', fn, epoch)

    except ValueError:
        # This can happen if the validation set only contains one class, making the confusion matrix non-2x2
        print('           - Could not calculate FP/FN. The validation set may contain only one class.\n')

    return avg_loss, accuracy


if __name__ == '__main__':
    args = get_args()
    train_model(args) 