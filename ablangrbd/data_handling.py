# defining_pub_clone/software/linear_classifier/data_handling.py
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer
# /mnt/hd2/clint/ml_results/linear_classifiers/ABLANG1plus6_16classes_125epochs_8batch_20240510_024812

# ablang_hc_hug_path = '/home/clint/.cache/huggingface/hub/models--qilowoq--AbLang_heavy/snapshots/ecac793b0493f76590ce26d48f7aac4912de8717/'
# ablang_lc_hug_path = '/home/clint/.cache/huggingface/hub/models--qilowoq--AbLang_light/snapshots/ce0637166f5e6e271e906d29a8415d9fdc30e377/'

ablang_hc_hug_path = '/dors/iglab/Members/holtcm/.cache/huggingface/hub/models--qilowoq--AbLang_heavy/snapshots/ecac793b0493f76590ce26d48f7aac4912de8717/'
ablang_lc_hug_path = '/dors/iglab/Members/holtcm/.cache/huggingface/hub/models--qilowoq--AbLang_light/snapshots/ce0637166f5e6e271e906d29a8415d9fdc30e377'
heavy_tokenizer = AutoTokenizer.from_pretrained(ablang_hc_hug_path)
light_tokenizer = AutoTokenizer.from_pretrained(ablang_lc_hug_path)

def get_starting_dataloaders_simple(run_specifics: dict):

    dataloader1 = DataLoader(torch.load("train_dataset.pt"), batch_size=run_specifics["TRAIN_BATCH_SIZE"], shuffle=True)
    dataloader2 = DataLoader(torch.load("test_dataset.pt"), batch_size=run_specifics["TEST_BATCH_SIZE"], shuffle=True)
    return dataloader1, dataloader2


def get_cos_sim_mat_cross(embeddings1: torch.Tensor, embeddings2: torch.Tensor, device) -> torch.Tensor:
    with torch.no_grad():
        # Calculate cosine similarity all vs all
        embeddings1, embeddings2 = embeddings1.to(device), embeddings2.to(device)
        cos_sim_mat = embeddings1 @ embeddings2.t()  # Get cosine similarities
        del embeddings1, embeddings2
        torch.cuda.empty_cache()

    return cos_sim_mat


def get_label_equalities_cross(labels1, labels2, device) -> torch.Tensor:
    labels1, labels2 = torch.tensor(labels1).to(device), torch.tensor(labels2).to(device)

    label_mat = torch.eq(labels1[:, None], labels2).to(torch.bool)  # Boolean
    return label_mat


def oversample_epitope(df: pd.DataFrame, epitope_col: str = "EPITOPE", group_col: str = '') -> pd.DataFrame:
    """
    Oversample the DataFrame so each EPITOPE category has the same 
    total number of rows, matching the maximum EPITOPE count.
    
    If 'group_col' is given, we distribute the new rows (the difference 
    to reach the max EPITOPE size) evenly across each subgroup within 
    that EPITOPE (with replacement).
    """
    # 1. Determine the max size among EPITOPEs
    epitope_counts = df[epitope_col].value_counts()
    max_count = epitope_counts.max()
    
    def _oversample_single_epitope(epitope_df: pd.DataFrame) -> pd.DataFrame:
        """Oversample this one EPITOPE subset."""
        current_count = len(epitope_df)
        needed = max_count - current_count
        if needed <= 0:
            # This epitope is already at or above the max; just return it
            return epitope_df
        
        # -- Case A: No subgrouping
        if not group_col:
            # Simply sample the needed rows from the whole epitope
            sampled = epitope_df.sample(n=needed, replace=True)
            return pd.concat([epitope_df, sampled], ignore_index=True)
        
        # -- Case B: Distribute new rows among subgroups (e.g. CLONOTYPE)
        grouped_list = list(epitope_df.groupby(group_col))
        n_subgroups = len(grouped_list)
        
        # Evenly distribute 'needed' among subgroups
        base = needed // n_subgroups
        remainder = needed % n_subgroups
        
        # Start with the original rows
        oversampled_pieces = [epitope_df]
        
        # Loop over each subgroup, allocating 'base' plus possibly 1 extra if remainder
        for idx, (subgroup_val, subgroup_df) in enumerate(grouped_list):
            needed_from_this_subgroup = base + (1 if idx < remainder else 0)
            if needed_from_this_subgroup > 0:
                # Sample with replacement in case the subgroup doesn't have enough unique rows
                sampled_sub = subgroup_df.sample(n=needed_from_this_subgroup, replace=True)
                oversampled_pieces.append(sampled_sub)
        
        return pd.concat(oversampled_pieces, ignore_index=True)
    
    # 2. Group by EPITOPE and oversample each subset
    df_oversampled = (
        df.groupby(epitope_col, group_keys=False)
          .apply(_oversample_single_epitope)
          .reset_index(drop=True)
    )
    
    return df_oversampled


def simple_sampling(df: pd.DataFrame, run_specifics) -> pd.DataFrame:
    """Create the column "DATASET" and assign each antibody "TRAIN", "TEST", "VAL".
    Create a second column "SYNTHETIC" with either True or False for each

    Args:
        df (pd.DataFrame): DataFrame with unlabeled train/test/val antibodies

    Returns:
        df
    """
    train_fract = run_specifics["TRAIN_FRACT"]
    # Downsample to 200 random rows from each of these
    dms_df_train = df.groupby("EPITOPE").apply(lambda x: x.sample(frac=train_fract)).reset_index(drop=True)

    # Get remainder of cells and split evenly into test and validate
    dms_df_remaining_cells = df[~df["INDEX"].isin(dms_df_train["INDEX"])]
    dms_df_test = dms_df_remaining_cells.groupby("EPITOPE").apply(lambda x: x.sample(frac=0.5)).reset_index(drop=True)  # Get 50%
    dms_df_val = dms_df_remaining_cells[~dms_df_remaining_cells["INDEX"].isin(dms_df_test["INDEX"])]
    # Now get non-redundant validation set
    used_clones = set(dms_df_train["CLONOTYPE"].unique()).union(set(dms_df_test["CLONOTYPE"].unique()))
    dms_df_val = dms_df_val[~dms_df_val["CLONOTYPE"].isin(used_clones)]

    # Save the dataframe with DATASET column: train, test with no redundant clones, test clones redundant to the training set, and test clones redundant to test_nr
    df.loc[:, "DATASET"] = "TEST"
    df.loc[df["INDEX"].isin(dms_df_train["INDEX"]), "DATASET"] = "TRAIN"
    df.loc[df["INDEX"].isin(dms_df_val["INDEX"]), "DATASET"] = "VAL"
    df.loc["SYNTHETIC"] = 0
    return df


def balanced_sampling(df: pd.DataFrame, run_specifics: dict) -> pd.DataFrame:
    """
    Similar to 'simple_sampling' but oversamples the TRAIN set (by EPITOPE 
    and optionally by group_col). After oversampling, duplicate/synthetic rows 
    are labeled with SYNTHETIC=1, and the original row remains SYNTHETIC=0.

    Args:
        df (pd.DataFrame): The full dataset (must have columns "EPITOPE", "CLONOTYPE", "INDEX", etc.).
        train_fract (float): Fraction of each epitope to assign to TRAIN (e.g. 0.8).
        epitope_col (str): Column name for your epitope (e.g. "EPITOPE").
        group_col (str): Optional column name for distribution within epitope (e.g. "CLONOTYPE").
                         If empty, oversampling only balances epitope counts.

    Returns:
        pd.DataFrame: A new DataFrame with columns "DATASET" and "SYNTHETIC".
                      TRAIN is oversampled, TEST and VAL are left as-is.
    """
    train_fract, epitope_col, group_col = run_specifics["TRAIN_FRACT"], run_specifics["EPITOPE_COL"], run_specifics["GROUP_COL"]
    # 1. Sample TRAIN fraction per EPITOPE
    dms_df_train = df.groupby(epitope_col)\
                     .apply(lambda x: x.sample(frac=train_fract, random_state=None))\
                     .reset_index(drop=True)

    # 2. The remainder is for TEST/VAL splitting
    dms_df_remaining_cells = df[~df["INDEX"].isin(dms_df_train["INDEX"])]
    dms_df_test = dms_df_remaining_cells.groupby(epitope_col)\
                                        .apply(lambda x: x.sample(frac=0.5, random_state=None))\
                                        .reset_index(drop=True)
    dms_df_val = dms_df_remaining_cells[~dms_df_remaining_cells["INDEX"].isin(dms_df_test["INDEX"])]

    # Now get non-redundant validation set
    used_clones = set(dms_df_train["CLONOTYPE"].unique()).union(set(dms_df_test["CLONOTYPE"].unique()))
    dms_df_val = dms_df_val[~dms_df_val["CLONOTYPE"].isin(used_clones)]

    # 3. Oversample only the TRAIN subset
    #    (This will create duplicates for the underrepresented epitope categories.)
    train_balanced = oversample_epitope(dms_df_train, epitope_col=epitope_col, group_col=group_col)

    # 4. Label synthetic rows in TRAIN:
    #    "SYNTHETIC=0" for the first occurrence of a row (by 'INDEX'), 
    #    "SYNTHETIC=1" for duplicates introduced by oversampling.
    #    We'll rely on the fact that duplicates (with replacement) preserve the same 'INDEX'.
    train_balanced["SYNTHETIC"] = (
        train_balanced.duplicated(subset="INDEX", keep="first").astype(int)
    )

    # For TEST and VAL, we do not oversample, so set SYNTHETIC=0 for them
    dms_df_test["SYNTHETIC"] = 0
    dms_df_val["SYNTHETIC"] = 0

    # 5. Label the DATASET column: "TRAIN", "TEST", or "VAL".
    train_balanced["DATASET"] = "TRAIN"
    dms_df_test["DATASET"] = "TEST"
    dms_df_val["DATASET"] = "VAL"

    # 6. Concatenate everything into one final DataFrame
    df_final = pd.concat([train_balanced, dms_df_test, dms_df_val], ignore_index=True)

    return df_final


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

def get_dataloader(df: pd.DataFrame, batch_size: int, shuffle: bool=True, dataset_fname: str = ''):
    # Tokenize the chains
    h_tokens, l_tokens = tokenize_chains(heavy_tokenizer, light_tokenizer, df)
    
    # Prep the labels
    labels = torch.tensor(df.loc[:, "EPITOPE_LABELS"].values, dtype=torch.int8)

    # Put them all into one dataset
    dataset = TensorDataset(h_tokens['input_ids'].to(torch.int8), h_tokens['attention_mask'].to(torch.int8),
                l_tokens['input_ids'].to(torch.int8), l_tokens['attention_mask'].to(torch.int8), labels)
    if dataset_fname:
        torch.save(dataset, dataset_fname)
    
    # Prep it for iterating over it
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def setup_data(run_specifics):
    # Load data
    input_fname, output_folder, test_batch_size = run_specifics['DATA_FNAME'], run_specifics['OUTPUT_FOLDER'], run_specifics['TEST_BATCH_SIZE']
    dms_df = pd.read_excel(input_fname)
    dms_df["INDEX"] = dms_df.index
    dms_df = dms_df.drop_duplicates(subset=["HC_AA", "LC_AA"]).sample(frac=1)

    # Prepare the Sequences for Tokenization

    dms_df.loc[:, "PREPARED_HC_SEQ"] = dms_df["HC_AA"].apply(lambda x: " ".join(list(x)))  # Add spaces between each amino acid
    dms_df.loc[:, "PREPARED_LC_SEQ"] = dms_df["LC_AA"].apply(lambda x: " ".join(list(x)))

    # Split dataset into held out and non-heldout epitopes if desired
    if run_specifics['HELD_OUT_EPITOPES'] is not None:
        held_out_df = dms_df[dms_df["EPITOPE"].isin(run_specifics['HELD_OUT_EPITOPES'])]
    if run_specifics['EPS_TO_KEEP'] == "ALL":
        eps_to_keep = list(sorted(dms_df["EPITOPE"].unique()))
    else:
        eps_to_keep = run_specifics['EPS_TO_KEEP']
        dms_df = dms_df[dms_df["EPITOPE"].isin(eps_to_keep)]

    if run_specifics["USE_4CLASSES"]:
        # Add numerical labels for ml
        eps_to_keep = ["CLASS1", "CLASS2", "CLASS3", "CLASS4"]
        epitope_mapper = dict(zip(eps_to_keep, range(len(eps_to_keep))))
        dms_df.loc[:, "EPITOPE_LABELS"] = dms_df["CLASS"].map(epitope_mapper)  # Number the categories 
    else:
        epitope_mapper = dict(zip(eps_to_keep, range(len(eps_to_keep))))
        dms_df.loc[:, "EPITOPE_LABELS"] = dms_df["EPITOPE"].map(epitope_mapper)  # Number the categories 
    
    nlabels = len(eps_to_keep)
    
    # Split into train/test/val and potentially add synthetic data
    sampler = run_specifics["SAMPLING_METHOD"]
    dms_df = sampler(dms_df, run_specifics)

    dms_df.to_excel(f"{run_specifics['OUTPUT_FOLDER']}/cur_ml_split_{run_specifics['runid']}.xlsx", index=False)

    # tokenize the sequences
    f_tr = dms_df["DATASET"] == "TRAIN"
    f_te = dms_df["DATASET"] == "TEST"
    f_va = dms_df["DATASET"] == "VAL"

    print("About to start Tokenizing")
    train_dataloader = get_dataloader(dms_df[f_tr], nlabels, run_specifics["TRAIN_BATCH_SIZE"], True, f"{output_folder}/train_dataset_{run_specifics['runid']}.pt")
    test_dataloader = get_dataloader(dms_df[f_te], nlabels, test_batch_size, True, f"{output_folder}/test_dataset_{run_specifics['runid']}.pt")
    # This line saves the dataset which is useful
    val_dataloader = get_dataloader(dms_df[f_va], nlabels, test_batch_size, True, f"{output_folder}/val_dataset_{run_specifics['runid']}.pt")
    print("Train/Test/Val done tokenizing")

    # Save the held out ones as well
    if run_specifics['HELD_OUT_EPITOPES'] is not None:
        if run_specifics["USE_4CLASSES"]:
            ntot_labels = 4
            held_out_df.loc[:, "EPITOPE_LABELS"] = held_out_df["CLASS"].map(epitope_mapper)
        else:
            ntot_labels = nlabels + len(run_specifics['HELD_OUT_EPITOPES'])
            held_out_mapper = dict(zip(run_specifics['HELD_OUT_EPITOPES'], range(nlabels, ntot_labels)))
            held_out_df.loc[:, "EPITOPE_LABELS"] = held_out_df["EPITOPE"].map(held_out_mapper)

        # The dataset is saved to file
        print("Saving held out dataset")
        held_out_dataloader = get_dataloader(held_out_df, ntot_labels, test_batch_size, True, f"{output_folder}/held_out_dataset_{run_specifics['runid']}.pt")

    
    return train_dataloader, test_dataloader