# This file is intentionally kept minimal to avoid circular imports
# and to prevent loading modules with hardcoded paths.

# Expose only the specific model class we need for fine-tuning.
from .model import BertForSequenceClassification