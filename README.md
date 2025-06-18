# Glucose Time-Series Prediction for Prediabetes

This project uses an encoder-only Transformer model to predict whether a patient is prediabetic based on their Continuous Glucose Monitor (CGM) time-series data.

## How it Works

The model takes a 48-hour window of glucose readings and classifies it as belonging to a "prediabetic" or "not prediabetic" patient.

### Data Preprocessing

1.  **Ingestion**: Reads a standard CSV file with `Subject`, `Date`, `Time`, `Gl` (Glucose), and `Label` columns.
2.  **Standardization**: The raw data is transformed into a standardized internal format: `ID`, `time`, `reading`, `label`.
3.  **Patient-Aware Splitting**: To prevent data leakage, the data is split into training and validation sets based on `Subject` ID. This ensures that all data from a single patient belongs to only one set.
4.  **Normalization**: Glucose readings are normalized using `sklearn.StandardScaler`. The scaler is **fit only on the training data** to prevent the model from getting information about the validation set's distribution.
5.  **Feature Engineering**: Each time step in a sequence has two features:
    *   The normalized glucose reading.
    *   The time of day (normalized to a 0-1 range).
6.  **Windowing**: The full time-series for each patient is sliced into 48-hour windows, with a new window starting every 6 hours (a sliding window). This serves as a powerful form of data augmentation.

### Model Architecture

The model is an **Encoder-Only Transformer** built for sequence classification.

1.  **Input Embedding**: A linear layer projects the 2-dimensional input features (`reading`, `time_of_day`) into the model's higher-dimensional space (`d_model`).
2.  **CLS Token**: A special, learnable `[CLS]` (classification) token is prepended to the start of each input sequence.
3.  **Positional Encoding**: Standard sinusoidal positional encodings are added to the sequence to give the model information about the order of the data points.
4.  **Transformer Encoder**: The sequence is processed by a stack of Transformer Encoder layers. The self-attention mechanism allows the model to weigh the importance of different data points in the 48-hour window when making a prediction.
5.  **Classification Head**: The final, contextualized representation of the `[CLS]` token is passed through a single linear layer to produce the classification logits for the two classes ("prediabetic" or "not prediabetic").

## How to Use

### 1. Installation

First, clone the repository and set up a Python virtual environment.

```bash
# Clone this repository
git clone <your-repo-url>
cd glucose

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data (Optional)

The repository includes a sample `cgm_data.csv`. You can regenerate it by running:
```bash
python -m src.data_generation
```

### 3. Train the Model

To train the model with default hyperparameters, run:
```bash
python -m src.train
```

You can easily customize the training process with command-line arguments. For example:
```bash
python -m src.train --epochs 20 --lr 0.0005 --batch_size 64 --run_name "new_experiment"
```
To see all available options, run:
```bash
python -m src.train --help
```

### 4. Visualize Training with TensorBoard

Training progress (loss, accuracy, etc.) is logged to the `runs/` directory. To view these logs, launch TensorBoard:

```bash
# Make sure your virtual environment is activated
tensorboard --logdir=runs
```
Then open your web browser to `http://localhost:6006/`.

### 5. Plot Patient Data

To visualize the glucose trace for a single patient from the CSV file:

```bash
python -m src.visualization
```
The script will list the available patient IDs and plot the data for the first one.

## Project Structure

- `src/`: Contains the source code for data processing, model definition, training, and evaluation.
- `data/`: Will contain the dataset (not checked into git).
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA).
- `requirements.txt`: A list of python dependencies for this project.

## Goal

The primary goal is to build a model that takes a time-series of glucose readings and outputs a binary classification (prediabetic or not). 