"""
Prepare the raw embeddings dataset
Single expected argument: path to the saved torch 2d array. 
Shape: (number_of_embeddings, embedding_dimensions)
"""
import os
import sys
import pickle
import torch
import requests
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python script.py <argument>")
    sys.exit(1)
input_file_path = sys.argv[1]
data_tensor = torch.load(input_file_path)

print(f"Input tensor: {input_file_path}")
n, n_embd = data_tensor.shape
print('Number of datapoints:', n, "Dimensions:", n_embd)

# Define the lengths of the training and validation sets
train_length = int(0.9 * n)  # 90% for training
valid_length = n - train_length  # 10% for validation

train_tensor = data_tensor[:train_length]
val_tensor = data_tensor[train_length:]

print(train_tensor.shape)  # This should show the shape of the training set
print(val_tensor.shape)  # This should show the shape of the validation set

torch.save(train_tensor, os.path.join(os.path.dirname(__file__), 'train.pt'))
torch.save(val_tensor, os.path.join(os.path.dirname(__file__), 'val.pt'))

# save the meta information as well, to help us encode/decode later
meta = {
    'headless': True,
    'n_embd': n_embd
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# Input tensor: ../data/data_tensor.pt
# Number of datapoints: 373599 Dimensions: 48
# torch.Size([336239, 48])
# torch.Size([37360, 48])
