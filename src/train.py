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
    
    # Standardize the dataframe
    df = pd.DataFrame()
    df['time'] = pd.to_datetime(df_raw['Time'])
    df['ID'] = df_raw['Subject']
    df['reading'] = df_raw['Gl']
    df['label'] = df_raw['Label'].map({'pre': 1, 'non': 0})
    
    # --- Stratified, Patient-Aware Splitting ---
    # Get unique patient IDs and their corresponding labels
    patient_labels = df[['ID', 'label']].drop_duplicates().set_index('ID')
    
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Data loaded. Train patients: {len(train_patient_ids)}, Validation patients: {len(val_patient_ids)}")
    print(f"Train samples (windows): {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- Model Initialization ---
    model = PrediabetesTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
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
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, DEVICE)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    print("\n--- Finished Training ---")
    writer.close()
    
    # Save the trained model (optional)
    # torch.save(model.state_dict(), f'{args.run_name}_model.pth')
    # print(f"Model saved to {args.run_name}_model.pth")


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_samples
    print(f'\nValidation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
    return avg_loss, accuracy


if __name__ == '__main__':
    args = get_args()
    train_model(args) 