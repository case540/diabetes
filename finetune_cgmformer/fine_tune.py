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

def main():
    """Main function to run the fine-tuning pipeline."""
    parser = argparse.ArgumentParser(description='Fine-tune CGMformer for classification.')
    parser.add_argument('--checkpoint_path', type=str, default='cgmformer_ckpt/cgm_ckp/checkpoint-30000', help='Path to the pre-trained model checkpoint.')
    parser.add_argument('--token_path', type=str, default='cgmformer_ckpt/cgm_ckp/token2id.pkl', help='Path to the tokenizer file.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--freeze_layers', type=int, default=6, help='Number of initial BERT layers to freeze (e.g., 6 out of 12).')
    parser.add_argument('--use-fake-data', action='store_true', help='Use generated fake data.')
    parser.add_argument('--data-path', type=str, default=None, help='Path to your real CGM data CSV file.')
    parser.add_argument('--no_interpolation', action='store_true', help='Do not perform interpolation for missing data.')
    args = parser.parse_args()

    # --- Argument Validation ---
    if not args.use_fake_data and not args.data_path:
        parser.error("A data source is required. Please specify either --use-fake-data or provide a path with --data-path.")

    if args.use_fake_data and args.data_path:
        logging.warning("Both --use-fake-data and --data-path were provided. Prioritizing --data-path.")

    # --- Data Loading ---
    logging.info("Loading and processing data...")
    if args.data_path:
        logging.info(f"Loading real data from {args.data_path}")
        try:
            cgm_df = pd.read_csv(args.data_path)
            logging.info(f"Loaded {len(cgm_df)} readings for {cgm_df['Subject'].nunique()} patients.")
        except FileNotFoundError:
            logging.error(f"Error: The file specified could not be found at {args.data_path}")
            return
    elif args.use_fake_data:
        logging.info("Loading fake data from cgm_data.csv")
        cgm_df = pd.read_csv("cgm_data.csv")
        logging.info(f"Loaded {len(cgm_df)} readings for {cgm_df['Subject'].nunique()} fake patients.")

    # --- Standardize column names to lowercase for consistency ---
    cgm_df.columns = [col.lower() for col in cgm_df.columns]

    # --- Pre-validation and Filtering ---
    logging.info(f"Found initial labels in data: {cgm_df['label'].unique()}")
    valid_labels = ['pre', 'non']
    original_rows = len(cgm_df)
    cgm_df = cgm_df[cgm_df['label'].isin(valid_labels)]
    
    if len(cgm_df) < original_rows:
        logging.warning(f"Filtered out {original_rows - len(cgm_df)} rows with invalid labels. Keeping only rows with labels: {valid_labels}")
    
    if len(cgm_df) == 0:
        logging.error("No valid data remaining after filtering for 'pre' and 'non' labels. Cannot continue.")
        return

    # --- Data Preprocessing ---
    # Convert 'Time' to datetime and 'Gl' to numeric, dropping any rows that fail conversion
    cgm_df['time'] = pd.to_datetime(cgm_df['time'])
    cgm_df['gl'] = pd.to_numeric(cgm_df['gl'], errors='coerce')
    cgm_df = cgm_df.dropna(subset=['gl'])

    # The dataset requires a 'health_condition' column from a 'bio_df'. We create it here.
    # We map 'Label' (from the CSV) to 'health_condition' using a standard convention.
    label_map = {"pre": "pre-diabetes", "non": "healthy"}
    bio_df = pd.DataFrame({
        'subject': cgm_df['subject'].unique()
    })

    # Get the first label for each subject to determine their health condition
    first_labels = cgm_df.groupby('subject').first()['label']
    bio_df['health_condition'] = bio_df['subject'].map(first_labels).map(label_map)
    
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
        cgm_df=cgm_df, bio_df=bio_df, token2id_path=args.token_path,
        use_interpolation=(not args.no_interpolation), patient_ids=train_patient_ids,
        is_train=True
    )

    val_dataset = CGMformerFinetuneDataset(
        cgm_df=cgm_df, bio_df=bio_df, token2id_path=args.token_path,
        use_interpolation=(not args.no_interpolation), patient_ids=val_patient_ids,
        is_train=False
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
    main() 