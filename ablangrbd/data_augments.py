import pandas as pd

import pandas as pd
import numpy as np

import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer
# /mnt/hd2/clint/ml_results/linear_classifiers/ABLANG1plus6_16classes_125epochs_8batch_20240510_024812

output_folder = ''
# starting_folder = '/mnt/hd2/clint/ml_results/linear_classifiers/ABLANG1Mixed_16batch_12epitopes_125epochs_20250127c'
starting_folder = '/dors/iglab/Members/holtcm/defining_pub_clone/ml_results/RBD/linear_classifiers/ABLANG1_16_20250127c_CONTRASTIVE_STARTING_POINT'

train_batch_size = 512  # 2**9
test_batch_size = 512

ablang_hc_hug_path = '/dors/iglab/Members/holtcm/.cache/huggingface/hub/models--qilowoq--AbLang_heavy/snapshots/ecac793b0493f76590ce26d48f7aac4912de8717/'
ablang_lc_hug_path = '/dors/iglab/Members/holtcm/.cache/huggingface/hub/models--qilowoq--AbLang_light/snapshots/ce0637166f5e6e271e906d29a8415d9fdc30e377'

# ablang_hc_hug_path = '/home/clint/.cache/huggingface/hub/models--qilowoq--AbLang_heavy/snapshots/ecac793b0493f76590ce26d48f7aac4912de8717/'
# ablang_lc_hug_path = '/home/clint/.cache/huggingface/hub/models--qilowoq--AbLang_light/snapshots/ce0637166f5e6e271e906d29a8415d9fdc30e377/'
heavy_tokenizer = AutoTokenizer.from_pretrained(ablang_hc_hug_path)
light_tokenizer = AutoTokenizer.from_pretrained(ablang_lc_hug_path)


def tokenize_chains(heavy_tokenizer, light_tokenizer, df):
    h_tokens = heavy_tokenizer.batch_encode_plus(
        df["PREPARED_HC_SEQ"].tolist(), 
        add_special_tokens=True, 
        padding='longest', 
        return_tensors="pt",
        return_special_tokens_mask=True)
    l_tokens = light_tokenizer.batch_encode_plus(
        df["PREPARED_LC_SEQ"].tolist(), 
        add_special_tokens=True, 
        padding='longest',             
        return_tensors="pt",
        return_special_tokens_mask=True)
    return h_tokens, l_tokens


def get_dataloader(heavy_tokenizer, light_tokenizer, df: pd.DataFrame, batch_size: int, shuffle: bool=True, dataset_fname: str = ''):
    # Tokenize the chains
    h_tokens, l_tokens = tokenize_chains(heavy_tokenizer, light_tokenizer, df)
    
    # Prep the labels
    labels = torch.tensor(df.loc[:, "EPITOPE_LABELS"].values)

    # Put them all inpipi to one dataset
    dataset = TensorDataset(h_tokens['input_ids'].to(torch.int8), h_tokens['attention_mask'].to(torch.int8),
                            l_tokens['input_ids'].to(torch.int8), l_tokens['attention_mask'].to(torch.int8), labels)
    if dataset_fname:
        torch.save(dataset, dataset_fname)
    
    # Prep it for iterating over it
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # return dataloader


def main():
    labeled_df = pd.read_excel(f'{starting_folder}/cur_ml_split.xlsx')
    train_df = labeled_df[~labeled_df["SYNTHETIC"].astype(bool) & (labeled_df["DATASET"] == "TRAIN")]
    test_df = labeled_df[~labeled_df["SYNTHETIC"].astype(bool) & (labeled_df["DATASET"] == "TEST")]

    # train_dataloader = get_dataloader(heavy_tokenizer, light_tokenizer, train_df, train_batch_size, shuffle=True, dataset_fname=os.path.join(output_folder, 'train_dataset.pt'))
    # test_dataloader = get_dataloader(heavy_tokenizer, light_tokenizer, test_df, test_batch_size, shuffle=True, dataset_fname=os.path.join(output_folder, 'test_dataset.pt'))
    get_dataloader(heavy_tokenizer, light_tokenizer, train_df, train_batch_size, shuffle=True, dataset_fname=os.path.join(output_folder, 'train_dataset.pt'))
    get_dataloader(heavy_tokenizer, light_tokenizer, test_df, test_batch_size, shuffle=True, dataset_fname=os.path.join(output_folder, 'test_dataset.pt'))
main()