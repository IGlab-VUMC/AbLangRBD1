"""defining_pub_clone/software/linear_classifier/run_linear_classifier.py

This file is meant for training and inference for various linear classifiers based off antibody LLMs.
Previously my code was very hard to read and so the goal here is to divide between multiple files ad make it easier to read

My thoughts are this:


"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18
plt.rcParams['figure.constrained_layout.use'] = True
import pandas as pd

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# device = "cpu"
import torch.nn.functional as F

import time
from datetime import datetime
date_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get date for saving
import os
import sys
import shutil
import pickle
import typing as T
from glob import glob
from tqdm.auto import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.utils.data import DataLoader

import data_handling
import models
import analysis
from get_run_specifics import get_run_specifics


def training_loop_contrastive(model, train_dataloader, optimizer, test_dataloader, run_specifics: dict):
    num_epochs, output_folder = run_specifics['NUM_EPOCHS'], run_specifics['OUTPUT_FOLDER']
    full_df = pd.read_pickle('rbd_dataset_16-2_linearembeds.pd')
    full_df.drop(columns=["EMBEDDING"], inplace=True)
    analysis.embed_df(run_specifics, full_df, model, f"000_embedded.pd")

    criterion = models.ContrastiveLoss(temperature=run_specifics["TEMPERATURE"])
    evaluation_criterion = models.ContrastiveTrainTestLoss(temperature=run_specifics["TEMPERATURE"])

    train_loss, test_loss, train_test_loss = analysis.get_losses(run_specifics, model, train_dataloader, test_dataloader, evaluation_criterion)
    metrics = []
    metrics.append({"EPOCH": 0, "TRAIN_LOSS": train_loss, "TEST_LOSS": test_loss, "TRAIN_TEST_LOSS": train_test_loss})
    print(f"Epoch 0: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Train-Test Loss = {train_test_loss:.4f}")
    all_losses = []
    logf = open(os.path.join(output_folder, "log.txt"), "w")
    
    for epoch in tqdm(range(1, num_epochs + 1), desc='Epochs', position=0):
        model.train()
        
        batch_progress = tqdm(train_dataloader, desc=f'Epoch {epoch}', leave=False, position=1)
        
        for batch in batch_progress:
            h_seqs, h_mask, l_seqs, l_mask, labels = [b.to(device) for b in batch]            
            optimizer.zero_grad()
            
            # Get embeddings
            embeddings = model(h_input_ids=h_seqs, h_attention_mask=h_mask,
                             l_input_ids=l_seqs, l_attention_mask=l_mask)
            
            # Compute contrastive loss
            loss = criterion(embeddings, labels)
            cur_loss = loss.item()
            all_losses.append(cur_loss)
            
            loss.backward()
            optimizer.step()
            
            batch_progress.set_postfix({'loss': f'{cur_loss:.4f}'})
            
            del h_seqs, h_mask, l_seqs, l_mask, embeddings, loss
            torch.cuda.empty_cache()
        
        # Log epoch metrics

        train_loss, test_loss, train_test_loss = analysis.get_losses(run_specifics, model, train_dataloader, test_dataloader, evaluation_criterion)
        metrics.append({"EPOCH": epoch, "TRAIN_LOSS": train_loss, "TEST_LOSS": test_loss, "TRAIN_TEST_LOSS": train_test_loss})
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Train-Test Loss = {train_test_loss:.4f}")

        
        logf.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Train-Test Loss = {train_test_loss:.4f}")  

        if (epoch % 20 == 0) or epoch in [1, 3, 5, 10]:
            model_name = f"model-{epoch:03d}.pt"
            analysis.embed_df(run_specifics, full_df, model, f"{epoch:03d}_embedded.pd")
            model.cpu()
            torch.save(model.state_dict(), model_name)
            model = model.to(device)
            

        batch_progress.close()
    
    logf.close()
    return pd.DataFrame(metrics), all_losses


def final_save(run_specifics, metrics, all_losses, all_accs):
    output_folder, runid = run_specifics["OUTPUT_FOLDER"], run_specifics["RUN_ID"]
    # Pickle all three results metrics, all_losses, all_accs
    df = pd.DataFrame(metrics)
    df.to_csv(f'{output_folder}/per_epoch_results_{runid}_final.csv', index=False)
    with open(f'{output_folder}/per_step_loss_acc_{runid}_final.pkl', 'wb') as f:
        pickle.dump((all_losses, all_accs), f)


def main(run_specifics):
    modelf = run_specifics["STARTING_MODEL_FILE"]
    model = models.setup_contrastive_model(run_specifics, modelf).to(device)
    optimizer = models.setup_optimizer(model, run_specifics)  # Set up the optimizer

    train_dataloader, test_dataloader = data_handling.get_starting_dataloaders_simple(run_specifics)  # Set up the data

    metrics, losses = training_loop_contrastive(model, train_dataloader, optimizer, test_dataloader, run_specifics)
    of = run_specifics["OUTPUT_FOLDER"]
    with open(os.path.join(of, 'all_train_losses.pkl'), "wb") as f:
        pickle.dump(losses, f)
    metrics.to_csv(os.path.join(of, 'final_metrics.csv'), index=False)
    # best_epoch = analysis.plot_metrics(metrics, all_losses, all_accs, f"{run_specifics['OUTPUT_FOLDER']}/per_epoch_line_plots")  # Plot the metrics
    # final_save(run_specifics, metrics, all_losses, all_accs)  # Rename files and remove unneeded ones

    # best_epoch_dict = metrics[metrics["EPOCH"] == best_epoch].iloc[0].to_dict()
    # return best_epoch_dict, new_model_name


if __name__ == "__main__":
    run_key = "SimCLR2_250129f"
    run_specifics = get_run_specifics(run_key)

    main(run_specifics)
    # train_test_best_epoch_dict, new_model_name = main(run_specifics)

    # for batch_size in [8, 16, 4, 32, 64]:
    #     for i in range(5):
    #         run_specifics["RUN_ID"] = f"batchsize{batch_size}-{i}"
    #         main(run_specifics)
            # train_test_best_epoch_dict, new_model_name = main(run_specifics)
    # train_test_results = []
    # val_results = []
    # for batch_size in [8]:
    # for batch_size in [4, 8, 16, 32, 64]:
    # for batch_size in [64, 32]:
        # run_specifics["TRAIN_BATCH_SIZE"] = batch_size
        # for i in range(5):
        # for i in range(2):
        #     run_specifics["RUN_ID"] = f"batchsize{batch_size}-{i}"
        #     # Define run parameters and set up for saving results
        #     date_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get date for saving
        #     # num_epochs = 1
        #     run_dict = {"BATCH_SIZE": batch_size, "RUN": i, "OUTPUT_FOLDER": output_folder, "NUM_CLASSES": run_specifics["NUM_CLASSES"]}

        #     # * Actually Perform the run ********************************
        #     train_test_best_epoch_dict, new_model_name = main(run_specifics)
            
        #     # Save the results to pandas data frames
        #     run_dict["BEST_EPOCH"] = int(train_test_best_epoch_dict.pop('EPOCH'))
        #     train_test_dict = run_dict.copy()
        #     train_test_dict.update(train_test_best_epoch_dict)
        #     train_test_results.append(train_test_dict)

        #     val_dict = run_dict.copy()
        #     val_best_epoch_dict = analysis.evaluate_validation_set(run_specifics, new_model_name)
        #     val_dict.update(val_best_epoch_dict)
        #     val_results.append(val_dict)

            # Plot Precision-Recall and ROC curves and save images (model_path: str, tensor_datasetf: TensorDataset, fig_name: str="")
            # analysis.plot_tradeoff_curves(model_path=new_model_name,
            #     tensor_datasetf=f"{output_folder}/test_dataset.pt", fig_name=f"{output_folder}/test_prec-rec_roc")
            # analysis.plot_tradeoff_curves(model_path=new_model_name,
            #     tensor_datasetf=f"{output_folder}/val_dataset.pt", fig_name=f"{output_folder}/val_prec-rec_roc")

    # with open(f"{output_folder}/train_test_val_results_tuple.pkl", "wb") as f:
    #     pickle.dump((train_test_results, val_results), f)  
    # # Write the full dataframes to tsv files
    # f = os.path.dirname(output_folder)
    # train_test_results_df = pd.DataFrame(train_test_results, columns=["BATCH_SIZE", "RUN", "NUM_CLASSES", "BEST_EPOCH", "TRAIN_LOSS", "TRAIN_ACC", "TRAIN_F1", "TEST_LOSS",  "TEST_ACC",  "TEST_F1", "OUTPUT_FOLDER"])
    # val_results_df = pd.DataFrame(val_results, columns=["BATCH_SIZE", "RUN", "NUM_CLASSES", "BEST_EPOCH", "VAL_LOSS", "VAL_ACC", "VAL_F1", "OUTPUT_FOLDER"])
    
    # for df, tsvf in [(train_test_results_df, f"{f}/ABLANG1_train-test_results_summary_{date_time}.tsv"),
    #                  (val_results_df,        f"{f}/ABLANG1_val_results_summary_{date_time}.tsv")]:
    #     # Average over all runs of the same batch size
    #     subset = df.drop(["OUTPUT_FOLDER", "BEST_EPOCH"], axis=1)
    #     avg = subset.groupby(["BATCH_SIZE"]).mean().reset_index()
    #     avg["RUN"] = "AVG"
    #     stdev = subset.groupby(["BATCH_SIZE"]).std().reset_index()
    #     stdev["RUN"] = "STDEV"
    #     df = pd.concat([df, avg, stdev], ignore_index=True)
    #     df.to_csv(tsvf, sep="\t")