"""
1/29/25 Seeing if I can embed the dataset with this code
"""

import os
import pickle
import pandas as pd
import torch

from transformers import AutoTokenizer

ablang_hc_hug_path = '/dors/iglab/Members/holtcm/.cache/huggingface/hub/models--qilowoq--AbLang_heavy/snapshots/ecac793b0493f76590ce26d48f7aac4912de8717/'
ablang_lc_hug_path = '/dors/iglab/Members/holtcm/.cache/huggingface/hub/models--qilowoq--AbLang_light/snapshots/ce0637166f5e6e271e906d29a8415d9fdc30e377'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# trained_model_folder = '/dors/iglab/Members/holtcm/defining_pub_clone/ml_results/RBD/linear_classifiers/ABLANG1_16_20250127c_CONTRASTIVE_STARTING_POINT'
best_model_fname = 'model_batchsize16-2_loss_model_epoch_093.pt'
batch_size = 528
# model_fn = os.path.join(trained_model_folder, best_model_fname)
model_fn = best_model_fname


# Step 1: Load the trained model
# os.chdir(trained_model_folder)
# print(os.getcwd())
import models
import analysis
import data_handling
from get_run_specifics import get_run_specifics

def embed(model, dataloader):
    model.eval()
    all_embeddings = []    
    for batch in dataloader:
        h_seqs, h_mask, l_seqs, l_mask = [b.to(device) for b in batch[:-1]]        
        with torch.no_grad():
            logits, embeddings = model(h_input_ids=h_seqs, h_attention_mask=h_mask, 
                                       l_input_ids=l_seqs, l_attention_mask=l_mask, 
                                       return_embedding=True)
            # Fill the pre-allocated tensors
            all_embeddings.extend(embeddings.cpu().numpy())
            del embeddings, logits, h_seqs, h_mask, l_seqs, l_mask
    return all_embeddings

heavy_tokenizer = AutoTokenizer.from_pretrained(ablang_hc_hug_path)
light_tokenizer = AutoTokenizer.from_pretrained(ablang_lc_hug_path)

# Put together everything I'll need
run_key = "ablang1_250127c"
run_specifics = get_run_specifics(run_key)

df = pd.read_pickle("rbd_dataset_16-2_split.pd")



dataloader = data_handling.get_dataloader(heavy_tokenizer, light_tokenizer, df, 16, batch_size=batch_size, shuffle=False)

model = models.setup_model(run_specifics, modelf=best_model_fname).to(device)
all_embeddings = embed(model, dataloader)

with open("all_embeddings.pkl", "wb") as f:
    pickle.dump(all_embeddings, f)

df.loc[:, "EMBEDDING"] = all_embeddings
df.to_pickle("rbd_dataset_16-2_split_embedded.pd")
# model = models.setup_model(run_specifics, modelf=model_fn).to(device)

# train_dataset = torch.