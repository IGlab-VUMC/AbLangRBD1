import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc

import typing as T
import os
from glob import glob
import pickle

import models
import data_handling


def get_losses(run_specifics, model, dataloader1, dataloader2, criterion):
    model = model.to(device)
    embedding_size = run_specifics["EMBEDDING_SIZE"]
    l1, l2 = len(dataloader1.dataset), len(dataloader2.dataset)
    total_samples = l1 + l2
    all_embeddings = torch.zeros((total_samples, embedding_size), dtype=torch.float32, device='cpu')
    all_labels = torch.zeros(total_samples, dtype=torch.int8, device='cpu')
    current_idx = 0
    for dataloader in [dataloader1, dataloader2]:
        for batch in dataloader:
            with torch.no_grad():
                h_seqs, h_mask, l_seqs, l_mask, labels = [b.to(device) for b in batch]
                batch_size = h_seqs.size(0)
                embeddings = model(h_input_ids=h_seqs, h_attention_mask=h_mask,
                                l_input_ids=l_seqs, l_attention_mask=l_mask)
                all_embeddings[current_idx:current_idx + batch_size] = embeddings.cpu()
                all_labels[current_idx:current_idx + batch_size] = labels.cpu()
        
    # Compute contrastive loss
    train_mask = torch.cat([torch.ones(l1), torch.zeros(l2)]).to(torch.bool)
    train_loss, test_loss, train_test_loss = criterion(all_embeddings, all_labels, train_mask)
    return train_loss.cpu().item(), test_loss.cpu().item(), train_test_loss.cpu().item()

def evaluate(model, dataloader, logf=None, dataset: str = "", embedding_size=1536):
    """ Evaluate model on dataset for average accuracy and loss
    """
    model.eval()
        # Calculate total size needed
    total_samples = len(dataloader.dataset)
    
    # Pre-allocate tensors
    all_preds = torch.zeros(total_samples, dtype=torch.long, device='cpu')
    all_labels = torch.zeros(total_samples, dtype=torch.long, device='cpu')
    all_losses = torch.zeros(total_samples, dtype=torch.float32, device='cpu')
    all_embeddings = torch.zeros((total_samples, embedding_size), dtype=torch.float32, device='cpu')
    
    # Use an index to keep track of where we are in the pre-allocated tensors
    current_idx = 0
    
    for batch in dataloader:
        h_seqs, h_mask, l_seqs, l_mask, labels = [b.to(device) for b in batch]
        batch_size = h_seqs.size(0)
        
        with torch.no_grad():
            logits, embeddings = model(h_input_ids=h_seqs, h_attention_mask=h_mask, 
                                       l_input_ids=l_seqs, l_attention_mask=l_mask, 
                                       return_embedding=True)
            
            preds = torch.argmax(logits, dim=1)
            real = torch.argmax(labels, dim=1)
            loss = F.cross_entropy(logits, labels, reduction='none')  # Get per-sample loss
            
            # Fill the pre-allocated tensors
            all_preds[current_idx:current_idx + batch_size] = preds.cpu()
            all_labels[current_idx:current_idx + batch_size] = real.cpu()
            all_losses[current_idx:current_idx + batch_size] = loss.cpu()
            all_embeddings[current_idx:current_idx + batch_size] = embeddings.cpu()
            
            current_idx += batch_size
            del embeddings, preds, labels, logits

    all_preds, all_labels, all_losses = all_preds.numpy(), all_labels.numpy(), all_losses.numpy()

    
    # Get per epitope metrics
    # per_lab_results_df = pd.DataFrame({"LABEL": list(range(6)), "NGUESS_TRUE": 0, "NGUESS_FALSE": 0, "NLABEL_MISSED": 0})

    # for guess, lab in zip(all_preds, all_labels):
    #     if guess == lab:
    #         per_lab_results_df.loc[lab, "NGUESS_TRUE"] += 1
    #     else:
    #         per_lab_results_df.loc[guess, "NGUESS_FALSE"] += 1
    #         per_lab_results_df.loc[lab, "NLABEL_MISSED"] += 1

    # Get average accuracy for guesses and labels
    f1 = f1_score(all_preds, all_labels, average='macro')
    # per_lab_results_df["GUESS_ACC"] = per_lab_results_df["NGUESS_TRUE"] / (per_lab_results_df["NGUESS_TRUE"] + per_lab_results_df["NGUESS_FALSE"])
    # per_lab_results_df["LABEL_ACC"] = per_lab_results_df["NGUESS_TRUE"] / (per_lab_results_df["NGUESS_TRUE"] + per_lab_results_df["NLABEL_MISSED"])
    # per_lab_results_df["LABEL"] = ["E2.2", "E3", "F1", "F2", "A", "C"]
    # print(per_lab_results_df)
    if logf is not None:
        logf.write(f"\n{dataset} F1 Score = {f1}\n")


    return accuracy_score(all_labels, all_preds), sum(all_losses) / len(all_losses), f1, all_embeddings, all_labels


def embed_df(run_specifics, df, model, output_name):
    if os.path.isfile("for_eval.pt"):
        dataloader = DataLoader(torch.load("for_eval.pt"), batch_size=run_specifics["TEST_BATCH_SIZE"], shuffle=False)
    else:
        dataloader = data_handling.get_dataloader(df, batch_size=run_specifics["TEST_BATCH_SIZE"], shuffle=False, dataset_fname="for_eval.pt")
    
    model.eval()
    all_embeddings = []    
    for batch in dataloader:
        h_seqs, h_mask, l_seqs, l_mask = [b.to(device) for b in batch[:-1]]        
        with torch.no_grad():
            embeddings = model(h_input_ids=h_seqs, h_attention_mask=h_mask,
                             l_input_ids=l_seqs, l_attention_mask=l_mask)
            # Fill the pre-allocated tensors
            all_embeddings.extend(embeddings.cpu().tolist())
            del embeddings, h_seqs, h_mask, l_seqs, l_mask

    df.loc[:, "EMBEDDING"] = all_embeddings
    df.to_pickle(output_name)
    df.drop(columns=["EMBEDDING"], inplace=True)
    return all_embeddings


def embed_dataset(run_specifics: dict, model: str, dataset_path: str, embedding_size: int=1536, batch_size: int=128):
    """Given a pytorch model and dataset, return the embeddings, predicted label, and actual label for each datapoint. 

    Args:
        model_path (str): Path to the model checkpoint
        dataset_path (str): Path to the dataset
        embedding_size (int): Size of the embedding vector. Defaults to 1536.
        batch_size (int): Batch size for processing. Defaults to 128.

    Returns:
        tuple: (embeddings, predictions, labels) where:
            - embeddings is a tensor of shape (n_samples, embedding_size)
            - predictions is a tensor of shape (n_samples,)
            - labels is a tensor of shape (n_samples,)
    """
    if type(model) == str:
        model = models.setup_model(run_specifics, modelf=model) # type: ignore
        
    model = model.to(device) # type: ignore
    model.eval() # type: ignore
    dataset = torch.load(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Pre-allocate tensors
    total_samples = len(dataset)
    all_preds = torch.zeros(total_samples, dtype=torch.int8, device='cpu')
    all_labels = torch.zeros(total_samples, dtype=torch.int8, device='cpu')
    all_embeddings = torch.zeros((total_samples, embedding_size), dtype=torch.float32, device='cpu')
    
    current_idx = 0
    for batch in dataloader:
        h_seqs, h_mask, l_seqs, l_mask, labels = [b.to(device) for b in batch]
        batch_size = h_seqs.size(0)  # Get actual batch size
        del batch
        
        with torch.no_grad():
            logits, embeddings = model(h_input_ids=h_seqs, 
                                     h_attention_mask=h_mask, 
                                     l_input_ids=l_seqs, 
                                     l_attention_mask=l_mask, 
                                     return_embedding=True) # type: ignore
            del h_seqs, h_mask, l_seqs, l_mask
            
            preds = torch.argmax(logits, dim=1).to(torch.int8)
            labels = torch.argmax(labels, dim=1).to(torch.int8)

            # Fill the pre-allocated tensors
            end_idx = current_idx + batch_size
            all_preds[current_idx:end_idx] = preds.cpu()
            all_labels[current_idx:end_idx] = labels.cpu()
            all_embeddings[current_idx:end_idx] = embeddings.cpu()

            current_idx = end_idx
            del embeddings, preds, labels, logits

    return all_embeddings, all_preds, all_labels


def evaluate_epoch(model: models.AbLang1LinearEmbedder, train_dataloader: DataLoader, test_dataloader: DataLoader,
                   metrics: list, epoch: int, logf: T.TextIO, run_specifics: dict, all_losses: T.List, all_accs: T.List):
    output_folder = run_specifics['OUTPUT_FOLDER']
    model.to(device)
    model.eval()

    # Do first set of evals
    train_dataloader2 = DataLoader(train_dataloader.dataset, batch_size=run_specifics["TEST_BATCH_SIZE"], shuffle=False)
    train_acc, train_loss, train_f1, train_embeds, train_labels = evaluate(model, train_dataloader2, logf, "Train")
    test_acc, test_loss, test_f1, test_embeds, test_labels = evaluate(model, test_dataloader, logf, "Test")

    # Get train vs test ROC and PW diff
    pw_diff = get_cross_dataset_pw_mean_diff(train_embeds, train_labels, test_embeds, test_labels)
    rocauc = get_cross_dataset_weighted_rocauc(train_embeds, train_labels, test_embeds, test_labels)


    # Store results
    metrics.append({"EPOCH": epoch, "TRAIN_LOSS": train_loss, "TRAIN_ACC": train_acc, "TRAIN_F1": train_f1,
                                    "TEST_LOSS":  test_loss,  "TEST_ACC":  test_acc,  "TEST_F1":  test_f1, "ROC-AUCS": rocauc, "PW-DIFFS": pw_diff})



    # Save results -- Don't access this prior to the first training epoch
    if len(all_losses) > 0:
        e = str(epoch).rjust(3, '0')
        model_save_method = run_specifics["MODEL_SAVE_METHOD"]
        model_save_method(model, e, run_specifics, metrics)
        # Every epoch save data and model as a checkpoint.
        if (epoch == 1) or (epoch % 10 == 0):
            with open(f"{output_folder}/results_epoch_{run_specifics['RUN_ID']}_{e}.pkl", 'wb') as f:
                pickle.dump((metrics, all_losses, all_accs), f)

            # Save the model
            # model.save_pretrained() # Save

        # plot_tsne(run_specifics, model, str(epoch), f"{output_folder}/train_dataset_{run_specifics['runid']}.pt", f"{run_specifics['RUN_ID']} Train Epoch {epoch}")
        # plot_tsne(run_specifics, model, str(epoch), f"{output_folder}/test_dataset_.pt", f"{run_specifics['RUN_ID']} Test Epoch {epoch}")


def evaluate_validation_set(run_specifics: dict, new_model_name: str):
    output_folder = run_specifics["OUTPUT_FOLDER"]
    model = models.setup_model(run_specifics, new_model_name).to(device)
    model.eval()
    dataloader = DataLoader(torch.load(f"{output_folder}/val_dataset_{run_specifics['runid']}.pt"), batch_size=128)

    val_acc, val_loss, val_f1, all_embeddings, all_labels = evaluate(model, dataloader)
    model = model.to("cpu")
    del model, dataloader

    with open(f"{output_folder}/val_acc.txt", "w") as f:
        f.write(f"Validation_Accuracy, Loss, F1\n{val_acc},{val_loss},{val_f1}\n")
    
    return {"VAL_LOSS": val_loss, "VAL_ACC": val_acc, "VAL_F1": val_f1}


def plot_metrics(metrics_df: pd.DataFrame, all_losses: list, all_accs: list, fig_name: str=""):
    """ Plot the metrics.

    
    :param fig_name (str, opt). If provided then save both a png and svg. Do not provide a file extension.
    """
    f = open(os.path.join(os.path.dirname(fig_name), "best_epoch_results.txt"), "w")
    # {"EPOCH": epoch, "TRAIN_LOSS": train_loss, "TRAIN_ACC": train_acc, "NORM_TRAIN_ACC": norm_train_acc,
    #  "TEST_LOSS":  test_loss,  "TEST_ACC":  test_acc,  "NORM_TEST_ACC":  norm_test_acc})
    # print(metrics_df)

    # Get best epoch for metrics
    ei = metrics_df['TEST_LOSS'].idxmax()
    e = metrics_df.loc[ei, 'EPOCH']
    print(f"Best Epoch (By Test loss): {e}")
    f.write(f"Best Epoch (By Test loss): {e}\n")
    l = metrics_df.loc[ei, 'TEST_LOSS']
    a = metrics_df.loc[ei, 'TEST_ACC']
    f1 = metrics_df.loc[ei, 'TEST_F1']
    print(f"Test Values: Loss {l:.4f}, Overall Accuracy {100*a:.2f}%, F1 Score {f1:.2f}") # type: ignore
    f.write(f"Test Values: Loss {l:.4f}, Overall Accuracy {100*a:.2f}%, F1 Score {f1:.2f}\n") # type: ignore
    l = metrics_df.loc[ei, 'TRAIN_LOSS']
    a = metrics_df.loc[ei, 'TRAIN_ACC']
    f1 = metrics_df.loc[ei, 'TRAIN_F1']
    print(f"Training Values: Loss {l:.4f}, Overall Accuracy {100*a:.2f}%, F1 Score {f1:.2f}") # type: ignore
    f.write(f"Training Values: Loss {l:.4f}, Overall Accuracy {100*a:.2f}%, F1 Score {f1:.2f}\n") # type: ignore

    # Plot per step metrics
    fig, ax = plt.subplots(2, 3, figsize=(20, 14))

    # Plot the loss values on the first subplot
    sns.lineplot(x='EPOCH', y='TRAIN_LOSS', data=metrics_df, ax=ax[0,0], label='Train Loss', color="blue")
    sns.lineplot(x='EPOCH', y='TEST_LOSS', data=metrics_df, ax=ax[0,0], label='Test Loss', color="orange")
    ax[0,0].set_title('Loss per Epoch')
    ax[0,0].set_xlabel('Epoch')
    ax[0,0].set_ylabel('Loss (Cross Entropy)')

    # Plot the accuracy values on the second subplot
    sns.lineplot(x='EPOCH', y='TRAIN_ACC', data=metrics_df, ax=ax[0,1], label='Overall Training Accuracy', color="blue")
    sns.lineplot(x='EPOCH', y='TEST_ACC', data=metrics_df, ax=ax[0,1], label='Overall Test Accuracy', color="orange")

    ax[0,1].set_title('Accuracy per Epoch')
    ax[0,1].set_xlabel('Epoch')
    ax[0,1].set_ylabel('Accuracy')

    # Plot the ROC-AUC values on the third subplot
    sns.lineplot(x='EPOCH', y="ROC-AUCS", data=metrics_df, ax=ax[0,2], label='Train-test ROC-AUC', color="blue")
    sns.lineplot(x='EPOCH', y="PW-DIFFS", data=metrics_df, ax=ax[0,2], label='Train-test ssab-dsab pw diff', color="orange")

    ax[0,2].set_title('Histogram metrics per Epoch')
    ax[0,2].set_xlabel('Epoch')
    ax[0,2].set_ylabel('ROC or pw diff')

    # Plot per step metrics
    ax[1, 0].plot(all_losses)
    ax[1, 0].set_title("Training Loss Per Step")
    ax[1, 0].set_xlabel("Step")
    ax[1, 0].set_ylabel("Loss (Cross Entropy)")

    ax[1, 1].plot(all_accs)
    ax[1,1].set_title("Training Accuracy Per Step")
    ax[1,1].set_xlabel("Step")
    ax[1,1].set_ylabel("Accuracy")

    # Save Figure if path given
    if fig_name:
        fig.savefig(fig_name + ".png", dpi=300)
        fig.savefig(fig_name + ".svg")

    # Show the plot
    plt.show()
    f.close()

    return e


def plot_tradeoff_curves(model_path: str, tensor_datasetf: str, fig_name: str=""):
    epitope_classes = ["E2.2", "E3", "F1", "F2", "A", "C"]
    # Load model and dataloader
    model = models.setup_model(model_path).to(device)
    model.eval()
    dataloader = DataLoader(torch.load(tensor_datasetf), batch_size=128)

    all_probs = []
    all_labels = []
    for batch in dataloader:
        h_seqs, h_mask, l_seqs, l_mask, labels = [b.to(device) for b in batch]
        with torch.no_grad():
            logits = model(h_input_ids=h_seqs, h_attention_mask=h_mask, l_input_ids=l_seqs, l_attention_mask=l_mask)
            probs = F.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Create a figure with 2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i, ep in enumerate(epitope_classes):
        # Compute precision-recall for each class and plot
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        ax[0].plot(recall, precision, lw=2)

        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        ax[1].plot(fpr, tpr, lw=2, label=f'Epitope {ep}')


    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")
    ax[0].set_title("Precision-Recall Curves")
    ax[0].grid(True)

    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("ROC Curves")
    ax[1].grid(True)
    ax[1].legend(loc="best")

    # Save Figure if path given
    if fig_name:
        fig.savefig(fig_name + ".png", dpi=300)
        fig.savefig(fig_name + ".svg")

    plt.show()


def plot_tsne(run_specifics: dict, model, e, dataset: str, title_info: str=''):
    output_folder = run_specifics["OUTPUT_FOLDER"]
    embeds, _, labels = embed_dataset(model, dataset, run_specifics["EMBEDDING_SIZE"], run_specifics["TEST_BATCH_SIZE"])
    label_set = ["CLASS1", "CLASS2", "CLASS3", "CLASS4"]
    label_map = dict(zip([0,1,2,3], label_set))
    all_labels = pd.Series(labels.numpy()).map(label_map).to_numpy()

    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=run_specifics["PERPLEXITY"])  # 30 seems good, 100 maybe even better. 5 probably too low
    tsne_reduction = tsne.fit_transform(embeds.numpy())

    x = tsne_reduction[:, 0]
    y = tsne_reduction[:, 1]
    # df.insert(0, col_name + "1", tsne_reduction[:, 0])
    # df.insert(1, col_name + "2", tsne_reduction[:, 1])

    # Visualize by Epitope Group
    plt.figure(figsize=(12, 8))
    plt.title("Embedding of RBD DMS Antibodies - " + title_info)
    # Define the color palette
    base_colors = sns.color_palette("husl", len(label_set))  # 7 colors with good contrast
    color_palette = {label: mcolors.rgb2hex(color) for label, color in zip(label_set, base_colors)}
    color_palette["Other"] = "#000000"  # Black for "Other"

    sns.scatterplot(
        x=x, y=y,
        hue=all_labels,
        palette=color_palette,
        legend="full"
    )
    plt.legend(title='RBD Classes', loc='center left', bbox_to_anchor=(1, 0.5))

    dataset_info = os.path.basename(dataset).split('_')[0]
    plt.savefig(f"{output_folder}/TSNE_epoch_{dataset_info}_{e}.png", dpi=300, bbox_inches='tight')  # TODO: make sure it is no longer 1 Gb


def get_cross_dataset_pw_mean_diff(embeddings1: torch.Tensor, labels1: np.ndarray,
                                   embeddings2: torch.Tensor, labels2: np.ndarray) -> float:
    """Take two embedding tensors from different datasets and get the mean difference between ssabs and dsabs

    Args:
        embeddings1 (torch.Tensor): embeddings from dataset 1, normalized (i.e. train)
        labels1 (torch.Tensor): labels from dataset 1, integers
        embeddings2 (torch.Tensor): embeddings from dataset 2, normalized (i.e. test)
        labels2 (torch.Tensor): labels from dataset 2, integers from same set as labels1

    Returns:
        float: the mean difference. Max is 2., min is -2.
    """
    cos_sim_mat = data_handling.get_cos_sim_mat_cross(embeddings1, embeddings2, device)
    label_mat = data_handling.get_label_equalities_cross(labels1, labels2, device)

    with torch.no_grad():
        # Get average of both same-label and different label pairs
        masked_cos_sim_mat = cos_sim_mat * label_mat
        same_label_avg = masked_cos_sim_mat.sum() / (label_mat).sum()

        masked_cos_sim_mat = cos_sim_mat * (~label_mat)
        diff_label_avg = masked_cos_sim_mat.sum() / (~label_mat).sum()

        mean_diff = float((same_label_avg - diff_label_avg).item())
        return mean_diff
    

def get_cross_dataset_weighted_rocauc(embeddings1: torch.Tensor, labels1: np.ndarray,
                                   embeddings2: torch.Tensor, labels2: np.ndarray) -> float:

    # Convert to numpy arrays
    embeddings1, embeddings2 = embeddings1.numpy(), embeddings2.numpy()
    labels1, labels2 = labels1, labels2
    
    # Get number of unique epitopes
    num_epitopes = int(max(labels1.max(), labels2.max()) + 1)  # Add 1 since 0-based indexing
    epitopes = list(range(num_epitopes))

    # Initialize arrays for metrics
    thresholds = np.linspace(-1, 1, 1001)
    tprs = np.zeros((num_epitopes, len(thresholds)))  # Same as recall
    fprs = np.zeros((num_epitopes, len(thresholds)))
    # precs = np.zeros((num_epitopes, len(thresholds)))
    ntrues = np.zeros(num_epitopes)  # Use this to weight

    for epitope_idx, epitope in enumerate(epitopes):
        # For each of these the dataset2, d2, (embeddings2 and label2) are not modified but d1 are

        # Use boolean indexing for subsetting
        ep_i_filt = labels1 == epitope
        d1_embeddings = embeddings1[ep_i_filt]
        d1_labels = labels1[ep_i_filt]
        
        # Calculate all comparisons at once
        e1e2_comparisons = d1_embeddings @ embeddings2.T

        # Reuse these comparisons for all thresholds
        actual_epitope_comparisons = (d1_labels[:, np.newaxis] == labels2)
        ntrue = actual_epitope_comparisons.sum()
        ntrues[epitope_idx] = ntrue
        nfalse = actual_epitope_comparisons.size - ntrue

        # Vectorized threshold calculations
        train_vs_val_guesses = e1e2_comparisons[:, :, np.newaxis] > thresholds
        
        true_positives = (train_vs_val_guesses & actual_epitope_comparisons[:, :, np.newaxis]).sum(axis=(0, 1))
        false_positives = (train_vs_val_guesses & ~actual_epitope_comparisons[:, :, np.newaxis]).sum(axis=(0, 1))
        
        tprs[epitope_idx] = true_positives / ntrue
        fprs[epitope_idx] = false_positives / nfalse
        # precs[epitope_idx] = true_positives / (true_positives + false_positives)

    # Calculate weights
    total_samples = len(embeddings1) + len(embeddings2)
    weights = ntrues / total_samples

    # Calculate weighted average TPRs and FPRs
    weighted_tprs = np.average(tprs, axis=0, weights=weights)
    weighted_fprs = np.average(fprs, axis=0, weights=weights)

    # Sort FPRs and TPRs for proper ROC curve
    sort_indices = np.argsort(weighted_fprs)
    weighted_fprs_sorted = weighted_fprs[sort_indices]
    weighted_tprs_sorted = weighted_tprs[sort_indices]

    # Calculate AUC
    weighted_auc = auc(weighted_fprs_sorted, weighted_tprs_sorted)
    return weighted_auc


