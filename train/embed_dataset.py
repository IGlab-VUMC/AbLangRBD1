import pandas as pd

from transformers import AutoTokenizer
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training 
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from get_run_specifics import get_run_specifics
import os

from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer
# /mnt/hd2/clint/ml_results/linear_classifiers/ABLANG1plus6_16classes_125epochs_8batch_20240510_024812

ablang_hc_hug_path = '/home/clint/.cache/huggingface/hub/models--qilowoq--AbLang_heavy/snapshots/ecac793b0493f76590ce26d48f7aac4912de8717/'
ablang_lc_hug_path = '/home/clint/.cache/huggingface/hub/models--qilowoq--AbLang_light/snapshots/ce0637166f5e6e271e906d29a8415d9fdc30e377/'
heavy_tokenizer = AutoTokenizer.from_pretrained(ablang_hc_hug_path)
light_tokenizer = AutoTokenizer.from_pretrained(ablang_lc_hug_path)

# os.chdir("SimCLR-2_202529f")
import models
import data_handling

modelf = "model-280.pt"
sab_rbds = pd.read_pickle("rbds_for_dms_analysis.pd")
if "EMBEDDING" in sab_rbds.columns:
    print("Yes")
    sab_rbds.drop(columns=["EMBEDDING"], inplace=True)
val_df = pd.read_pickle('280_embedded.pd')

run_key = "SimCLR2_250129f"
run_specifics = get_run_specifics(run_key)
# model =  models.setup_contrastive_model(run_specifics, modelf)
def setup_trained_model(run_specifics, modelf: str):
    """Setup a pre-trained LORA model with exact configuration matching"""
    
    # 1. Load the state dict first to examine its structure
    state_dict = torch.load(modelf)
    
    # 2. Create base model with correct class
    base_model = models.AbLangContrastive(add_mixer=run_specifics["MIXER"], use_cls=run_specifics["USE_CLS"])
    
    # 3. Convert the model to QLORA format BEFORE applying LORA config
    model = prepare_model_for_kbit_training(base_model)
    
    # 4. Create and apply LORA config
    target_modules = run_specifics["TARGET_MODULES"].copy()
    if run_specifics['MIXER']:
        target_modules.extend([f"mixer.layers.{i}" for i in range(0, 11, 2)])
        
    lora_config = LoraConfig(
        r=run_specifics["LORA_R"],
        lora_alpha=run_specifics["LORA_ALPHA"],
        target_modules=target_modules,
        lora_dropout=run_specifics["LORA_DROPOUT"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
        inference_mode=False
    )
    
    # 5. Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # 6. Remap the state dict keys to match LORA structure
    remapped_state_dict = {}
    for k, v in state_dict.items():
        if 'base_layer' in k:
            # Extract the core name without base_layer
            new_key = k.replace('.base_layer', '')
            remapped_state_dict[new_key] = v
    
    # 7. Load remapped state dict
    missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    
    # 8. Enable mixer parameters if present 
    if run_specifics['MIXER'] and hasattr(model, 'mixer'):
        for param in model.mixer.parameters():
            param.requires_grad = True
            
    return model.to(device)

import torch
import torch.nn as nn
from typing import Dict, Any
from peft import LoraConfig, get_peft_model

def setup_trained_model2(run_specifics: Dict[str, Any], modelf: str, device: torch.device) -> nn.Module:
    """
    Setup a pre-trained LoRA model with configuration matching training.
    
    Parameters:
        run_specifics: Configuration dictionary (should match training settings)
        modelf: Path to the saved state dict file.
        device: The torch device to which the model should be moved.
    
    Returns:
        The loaded model on the specified device.
    """
    # 1. Load the state dict (mapping to CPU)
    state_dict = torch.load(modelf, map_location="cpu")
    
    # 2. Create the base model and prepare for QLORA
    base_model = models.AbLangContrastive(
        add_mixer=run_specifics["MIXER"],
        use_cls=run_specifics["USE_CLS"]
    )
    model = prepare_model_for_kbit_training(base_model)
    
    # 3. Create and apply the LoRA config (inference_mode must match training, i.e. False)
    target_modules = run_specifics["TARGET_MODULES"].copy()
    if run_specifics["MIXER"]:
        target_modules.extend([f"mixer.layers.{i}" for i in range(0, 11, 2)])
        
    lora_config = LoraConfig(
        r=run_specifics["LORA_R"],
        lora_alpha=run_specifics["LORA_ALPHA"],
        target_modules=target_modules,
        lora_dropout=run_specifics["LORA_DROPOUT"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
        inference_mode=False  # ensure this matches training mode
    )
    model = get_peft_model(model, lora_config)
    
    # 4. Remap keys: Remove '.base_layer' from all keys in the state dict.
    remapped_state_dict = {k.replace('.base_layer', ''): v for k, v in state_dict.items()}
    
    # 5. Load the remapped state dict and print any discrepancies.
    missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    
    # 6. Re-enable mixer parameters if applicable.
    if run_specifics["MIXER"] and hasattr(model, "mixer"):
        for param in model.mixer.parameters():
            param.requires_grad = True
            
    return model.to(device)

# Usage:
model = setup_trained_model2(run_specifics, modelf, device)


def embed(model, dataloader):
    model.eval()
    all_embeddings = []    
    for batch in dataloader:
        h_seqs, h_mask, l_seqs, l_mask = [b.to(device) for b in batch]        
        with torch.no_grad():
            embeddings = model(h_input_ids=h_seqs, h_attention_mask=h_mask, 
                                       l_input_ids=l_seqs, l_attention_mask=l_mask, 
                                       return_embedding=True)
            # Fill the pre-allocated tensors
            all_embeddings.extend(embeddings.cpu().tolist())
            del embeddings, h_seqs, h_mask, l_seqs, l_mask
    return all_embeddings

def get_dataloader(df: pd.DataFrame, batch_size: int, shuffle: bool=True, dataset_fname: str = ''):
    # Tokenize the chains
    h_tokens, l_tokens = data_handling.tokenize_chains(heavy_tokenizer, light_tokenizer, df)
    
    # Put them all into one dataset
    dataset = TensorDataset(h_tokens['input_ids'].to(torch.int8), h_tokens['attention_mask'].to(torch.int8),
                l_tokens['input_ids'].to(torch.int8), l_tokens['attention_mask'].to(torch.int8))
    if dataset_fname:
        torch.save(dataset, dataset_fname)
    
    # Prep it for iterating over it
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

import analysis
# Put together everything I'll need
run_key = "SimCLR2_250129f"
run_specifics = get_run_specifics(run_key)

# analysis.embed_df(run_specifics, df, model, output_name)

dataloader_val = get_dataloader(val_df, 256, shuffle=False, dataset_fname="tempval.pt") 
# dataloader_val = get_dataloader(sab_rbds, 256, shuffle=False, dataset_fname="tempval.pt") 
all_embeddings = embed(model, dataloader_val)
# sab_rbds.loc[:, "EMBEDDING"] = all_embeddings
og_embeds = torch.tensor(val_df["EMBEDDING"].tolist())
new_embeds = torch.tensor(all_embeddings)
new_cos_sims = new_embeds @ new_embeds.t()
print(new_cos_sims.min(), new_cos_sims.mean(), new_cos_sims.max())
dist1 = torch.norm(og_embeds - new_embeds)
print(dist1)
row_diff = torch.norm(og_embeds - new_embeds, dim=1)

# Calculate the mean of the row-wise differences
mean_diff = row_diff.mean()

# Display the result
print("Mean row-wise Euclidean difference:", mean_diff.item())

# sab_rbds.loc[:, "EMBEDDING"] = all_embeddings
# sab_rbds.to_pickle("rbds_for_dms_analysis_maybe_embedded2.pd")


