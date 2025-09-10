import torch
from transformers import AutoTokenizer
from model import BERTAlignModel # Make sure this points to your model.py file

# --- Configuration ---
# 1. Path to your original PyTorch Lightning checkpoint file
CKPT_PATH = r"C:\Users\HP\PycharmProjects\PythonProject2\vero\metrics\align_score\alignscore\AlignScore-large.ckpt"

# 2. Path to the new folder where the standard Hugging Face model will be saved
SAVE_DIRECTORY = r"./alignscore_hf_model"

# --- Conversion Script ---
print("Loading model from .ckpt file...")
# Load the entire Lightning module from the checkpoint
lightning_model = BERTAlignModel.load_from_checkpoint(CKPT_PATH, strict=False)

# Extract the core transformer model from the Lightning wrapper
core_model = lightning_model.base_model

print(f"Saving model to Hugging Face format at: {SAVE_DIRECTORY}")
# Use the .save_pretrained() method to save it in the standard format
core_model.save_pretrained(SAVE_DIRECTORY)

print("Saving tokenizer...")
# Also save the appropriate tokenizer to the same directory
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
tokenizer.save_pretrained(SAVE_DIRECTORY)

print("✅ Conversion complete!")