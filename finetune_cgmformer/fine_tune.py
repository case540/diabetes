import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
import glob
import argparse
import logging
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report

# Add the cgmformer submodule to the Python path
sys.path.append('cgmformer')

from CGMFormer import BertForSequenceClassification
from .dataset import CGMformerFinetuneDataset

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_path='data/CGMacros_extracted', use_fake_data=False):
    """Loads and preprocesses data. Can use real CGMacros data or generated fake data."""
    if use_fake_data:
        logging.info("Loading fake data from cgm_data.csv")
        fake_data_path = 'cgm_data.csv'
        if not os.path.exists(fake_data_path):
            raise FileNotFoundError(f"Fake data file not found at {fake_data_path}. Please generate it first.")
        
        cgm_df = pd.read_csv(fake_data_path)
        cgm_df = cgm_df.rename(columns={'Subject': 'subject', 'Time': 'time', 'Gl': 'gl'})

        # Create a mock bio_df from the labels in the fake data
        bio_df = cgm_df[['subject', 'Label']].drop_duplicates().reset_index(drop=True)
        bio_df = bio_df.rename(columns={'Label': 'health_condition'})
        
        # Map labels to match the expected format ('pre' -> 'pre-diabetes', 'non' -> 'healthy')
        bio_df['health_condition'] = bio_df['health_condition'].replace({'non': 'healthy', 'pre': 'pre-diabetes'})
        
        logging.info(f"Loaded {len(cgm_df)} readings for {len(bio_df)} fake patients.")
        return cgm_df, bio_df

    # --- Existing logic for CGMacros data ---
    logging.info("Loading real data from CGMacros dataset.")
    base_path = ""
    for root, dirs, files in os.walk(data_path):
        if "bio.csv" in files:
            base_path = root
            break
    if not base_path:
        raise FileNotFoundError(f"Could not find bio.csv in {data_path}")

    bio_df = pd.read_csv(os.path.join(base_path, 'bio.csv'))
    def assign_health_condition(a1c):
        if a1c < 5.7: return 'healthy'
        elif 5.7 <= a1c <= 6.4: return 'pre-diabetes'
        else: return 't2d'
    bio_df['health_condition'] = bio_df['A1c PDL (Lab)'].apply(assign_health_condition)
    # bio_df = bio_df.rename(columns={'subject': 'subject_id'})

    # --- Filter for our 2-class problem (healthy vs. pre-diabetes) ---
    logging.info(f"Original dataset has {len(bio_df)} patients.")
    valid_conditions = ['healthy', 'pre-diabetes']
    bio_df_filtered = bio_df[bio_df['health_condition'].isin(valid_conditions)].copy()
    valid_subjects = bio_df_filtered['subject'].unique()
    logging.info(f"Filtering to {len(bio_df_filtered)} patients for 2-class problem (healthy/pre-diabetes).")

    cgm_files = sorted(glob.glob(os.path.join(base_path, 'CGMacros-*', 'cgm*.csv')))
    cgm_df_list = []
    for file in cgm_files:
        subject_id = int(os.path.basename(file).replace('cgm', '').replace('.csv', ''))
        temp_df = pd.read_csv(file, usecols=['time', 'gl'])
        temp_df['subject_id'] = subject_id
        cgm_df_list.append(temp_df)
    
    all_cgm_data = pd.concat(cgm_df_list, ignore_index=True)
    all_cgm_data = all_cgm_data.rename(columns={'subject_id': 'subject'})
    
    # Filter the CGM data to only include the subjects we're interested in
    all_cgm_data_filtered = all_cgm_data[all_cgm_data['subject'].isin(valid_subjects)]
    
    return all_cgm_data_filtered, bio_df_filtered

def main(args):
    # --- Load Data ---
    logging.info("Loading and processing data...")
    all_cgm_data, bio_df = load_data(use_fake_data=args.use_fake_data)
    
    # --- Stratified, Patient-Aware Train/Validation Split ---
    logging.info("Performing patient-aware stratified split.")
    patients_by_class = bio_df.groupby('health_condition')['subject'].apply(list)
    train_patient_ids, val_patient_ids = [], []

    for p_class, p_list in patients_by_class.items():
        np.random.shuffle(p_list)
        split_idx = int(0.8 * len(p_list))
        if split_idx == 0 and len(p_list) > 0: # Ensure at least one validation patient if possible
            split_idx = len(p_list) -1

        train_patient_ids.extend(p_list[:split_idx])
        val_patient_ids.extend(p_list[split_idx:])
    
    logging.info(f"Train patients: {len(train_patient_ids)}, Validation patients: {len(val_patient_ids)}")

    train_dataset = CGMformerFinetuneDataset(
        cgm_df=all_cgm_data, bio_df=bio_df, token2id_path=args.token_path,
        use_interpolation=(not args.no_interpolation), patient_ids=train_patient_ids
    )

    val_dataset = CGMformerFinetuneDataset(
        cgm_df=all_cgm_data, bio_df=bio_df, token2id_path=args.token_path,
        use_interpolation=(not args.no_interpolation), patient_ids=val_patient_ids
    )
    
    num_labels = len(train_dataset.label_map)
    logging.info(f"Task has {num_labels} classes.")

    # --- Train/Val Split ---
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- Load Pre-trained Model ---
    logging.info(f"Loading pre-trained model from {args.checkpoint_path}")
    model = BertForSequenceClassification.from_pretrained(
        args.checkpoint_path,
        num_labels=num_labels, # This will initialize a new classification head
    ).to(DEVICE)
    
    # --- Freeze Layers (Optional) ---
    if args.freeze_layers > 0:
        logging.info(f"Freezing the first {args.freeze_layers} encoder layers.")
        modules_to_freeze = model.bert.encoder.layer[:args.freeze_layers]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    # --- Training Setup ---
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # --- Fine-tuning Loop ---
    logging.info("Starting fine-tuning...")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids=sequences).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                outputs = model(input_ids=sequences).logits
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        report = classification_report(all_labels, all_preds, target_names=train_dataset.label_map.keys(), zero_division=0)
        
        logging.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"\nValidation Report for Epoch {epoch+1}:\n{report}\n")

    logging.info("Fine-tuning complete.")
    # Optional: Save the fine-tuned model
    # model.save_pretrained(f'finetuned_model_epoch_{args.epochs}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune CGMformer model.")
    parser.add_argument('--checkpoint_path', type=str, default='cgmformer_ckpt/cgm_ckp/checkpoint-30000', help='Path to the pre-trained model checkpoint.')
    parser.add_argument('--token_path', type=str, default='cgmformer_ckpt/cgm_ckp/token2id.pkl', help='Path to the token2id.pkl file.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to fine-tune.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--freeze_layers', type=int, default=6, help='Number of initial BERT layers to freeze (e.g., 6 out of 12).')
    parser.add_argument('--no_interpolation', action='store_true', help='Use simplified data processing without 5-min interval interpolation.')
    parser.add_argument('--use-fake-data', action='store_true', help='Use the generated cgm_data.csv instead of CGMacros.')
    
    args = parser.parse_args()
    main(args) 