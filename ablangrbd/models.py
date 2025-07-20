# defining_pub_clone/software/linear_classifier/models.py
"""
ML Models for OSAb prediction
"""
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training 

import pandas as pd

import torch
import torch.nn as nn

import torch.nn.functional as F

from time import time
import os
from glob import glob

import typing as T
# ablang_hc_hug_path = '/home/clint/.cache/huggingface/hub/models--qilowoq--AbLang_heavy/snapshots/ecac793b0493f76590ce26d48f7aac4912de8717/'
# ablang_lc_hug_path = '/home/clint/.cache/huggingface/hub/models--qilowoq--AbLang_light/snapshots/ce0637166f5e6e271e906d29a8415d9fdc30e377/'

ablang_hc_hug_path = '/dors/iglab/Members/holtcm/.cache/huggingface/hub/models--qilowoq--AbLang_heavy/snapshots/ecac793b0493f76590ce26d48f7aac4912de8717/'
ablang_lc_hug_path = '/dors/iglab/Members/holtcm/.cache/huggingface/hub/models--qilowoq--AbLang_light/snapshots/ce0637166f5e6e271e906d29a8415d9fdc30e377'


def setup_optimizer(model, run_specifics: dict):
    # Define the optimizer
    optimizer_fn, lr, scheduler_fn = run_specifics["OPTIMIZER"], run_specifics["LEARNING_RATE"], run_specifics["SCHEDULER"]
    optimizer = optimizer_fn(model.parameters(), lr)
    if scheduler_fn is not None:  # Fill in later if needed
        return optimizer, scheduler_fn
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    return optimizer


def setup_contrastive_model(run_specifics, modelf: str=""):
    """Setup model for contrastive learning with QLORA"""
    model = AbLangContrastive(add_mixer=run_specifics["MIXER"], use_cls=run_specifics["USE_CLS"])
    
    target_modules = run_specifics["TARGET_MODULES"]
    if run_specifics['MIXER']:
        target_modules.extend([f"mixer.layers.{i}" for i in range(0, 11, 2)])

    # Prepare the model for QLORA training
    if not modelf:
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=run_specifics["LORA_R"],
            lora_alpha=run_specifics["LORA_ALPHA"],
            target_modules=target_modules,
            lora_dropout=run_specifics["LORA_DROPOUT"],
            bias="none",
            task_type="FEATURE_EXTRACTION",  # Changed from SEQ_CLS since we're doing contrastive learning
        )

        model = get_peft_model(model, lora_config)

        if run_specifics['MIXER'] and hasattr(model, 'mixer'):
            for param in model.mixer.parameters():
                param.requires_grad = True

        model.print_trainable_parameters()

    else:
        # Load existing PEFT model
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=run_specifics["LORA_R"],
            lora_alpha=run_specifics["LORA_ALPHA"],
            target_modules=target_modules,
            lora_dropout=run_specifics["LORA_DROPOUT"],
            bias="none",
            task_type="FEATURE_EXTRACTION",  # Changed from SEQ_CLS since we're doing contrastive learning
        )
        model = get_peft_model(model, lora_config)
        state_dict = torch.load(modelf)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
        model.load_state_dict(state_dict, strict=False)
        
        # Re-enable training for MIXER if present
        if run_specifics['MIXER'] and hasattr(model, 'mixer'):
            for param in model.mixer.parameters():
                param.requires_grad = True

        model.print_trainable_parameters()

    return model


def setup_model(run_specifics, modelf: str=""):
    """
    :param modelf (str) If provided then load the state dict from the path.
    :return model (SiameseModel)
    """
    Model = run_specifics["MODEL_CLASS"]
    model = Model(add_mixer=run_specifics["MIXER"], use_cls=run_specifics["USE_CLS"], num_classes=run_specifics["NUM_CLASSES"])
    
    target_modules = run_specifics["TARGET_MODULES"]
    if run_specifics['MIXER']:
        target_modules.extend([f"mixer.layers.{i}" for i in range(0, 11, 2)]) # 1,179,648 -> 1,474,560 in both cases

    # Prepare the model for QLORA training
    if not modelf:
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
        r=run_specifics["LORA_R"],  # Reasonable values include 4,8,16,32. Determines number of trainable params. lower equals faster and fewer params
        lora_alpha=run_specifics["LORA_ALPHA"],
        target_modules=target_modules,
        lora_dropout=run_specifics["LORA_DROPOUT"],  # Was .05
        bias="none",
        task_type="SEQ_CLS",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # This should unfreeze the classifier
        # for param in model.classifier.parameters():
        #     param.requires_grad = True

        # model.print_trainable_parameters()

    else:
        model.load_state_dict(torch.load(modelf), strict=False)  # type: ignore # Without strict it could have issues

    return model


class ContrastiveTrainTestLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, epitope_labels, dataset_labels):
        """
        Args:
            embeddings: Normalized embeddings from the model
            epitope_labels: Integer tensor of epitope labels
            dataset_labels: Boolean tensor (True=train, False=test)
        Returns:
            train_loss: Loss computed only within training samples
            test_loss: Loss computed only within test samples
            cross_loss: Loss computed between train and test samples
        """
        with torch.no_grad():
            # Calculate similarity matrix
            sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
            
            # Create mask for positive pairs (same epitope)
            labels_matrix = epitope_labels.unsqueeze(0) == epitope_labels.unsqueeze(1)
            
            # Create masks for train and test samples
            train_mask = dataset_labels.unsqueeze(0) & dataset_labels.unsqueeze(1)  # train-train pairs
            test_mask = (~dataset_labels).unsqueeze(0) & (~dataset_labels).unsqueeze(1)  # test-test pairs
            cross_mask = (dataset_labels.unsqueeze(0) & (~dataset_labels).unsqueeze(1)) | \
                        ((~dataset_labels).unsqueeze(0) & dataset_labels.unsqueeze(1))  # train-test pairs
            
            # Remove self-comparisons            
            diag_mask = ~torch.eye(labels_matrix.shape[0], dtype=bool, device=labels_matrix.device)
            train_mask = train_mask & diag_mask
            test_mask = test_mask & diag_mask

            # Create positive pair masks for each subset
            train_positives = labels_matrix & train_mask
            test_positives = labels_matrix & test_mask
            cross_positives = labels_matrix & cross_mask
            
            # For numerical stability
            sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - sim_max.detach()
            
            # Compute exp of similarities
            exp_sim = torch.exp(sim_matrix)
            
            def compute_loss_for_mask(positive_mask, sample_mask):
                """Helper function to compute loss for a specific mask"""
                if positive_mask.sum() == 0:  # If no positive pairs in this mask
                    return torch.tensor(0.0, device=embeddings.device)
                    
                # Get denominators for the relevant samples
                # Only include pairs within the sample_mask in denominator
                exp_sim_masked = exp_sim * sample_mask
                denominator = exp_sim_masked.sum(dim=1)
                
                # Get positive pairs
                positives = torch.masked_select(exp_sim, positive_mask)
                # Repeat denominators for each positive pair
                denominators = denominator.repeat_interleave(positive_mask.sum(dim=1))
                
                # Compute loss
                losses = -torch.log(positives / denominators)
                return losses.mean() if losses.numel() > 0 else torch.tensor(0.0, device=embeddings.device)
            
            # Compute three types of losses
            train_loss = compute_loss_for_mask(train_positives, train_mask)
            test_loss = compute_loss_for_mask(test_positives, test_mask)
            cross_loss = compute_loss_for_mask(cross_positives, cross_mask)
            
            return train_loss, test_loss, cross_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, epitope_labels):
        """
        Args:
            embeddings: Normalized embeddings from the model
            epitope_labels: Integer tensor of epitope labels
        """
        # Calculate similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same epitope)
        labels_matrix = epitope_labels.unsqueeze(0) == epitope_labels.unsqueeze(1)
        mask_positives = labels_matrix & (~torch.eye(labels_matrix.shape[0], dtype=bool, 
                                                   device=labels_matrix.device))
        
        # For numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        
        # Compute exp of similarities
        exp_sim = torch.exp(sim_matrix)
        
        # Calculate denominator (sum over all non-self pairs)
        exp_sim_no_self = exp_sim * (~torch.eye(exp_sim.shape[0], dtype=bool, device=exp_sim.device))
        denominator = exp_sim_no_self.sum(dim=1)
        
        # Calculate loss for all positive pairs
        positives = torch.masked_select(exp_sim, mask_positives)
        denominators = denominator.repeat_interleave(mask_positives.sum(dim=1))
        
        # Compute final loss
        losses = -torch.log(positives / denominators)
        return losses.mean()


class Mixer(nn.Module):
    def __init__(self, in_d: int=1536):
        super(Mixer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_d, in_d), # First layer
            nn.ReLU(),             # First activation function
            nn.Linear(in_d, in_d), # Second layer
            nn.ReLU(),             # Second activation function
            nn.Linear(in_d, in_d), # Third layer
            nn.ReLU(),             # Third activation function
            nn.Linear(in_d, in_d), # Fourth layer
            nn.ReLU(),             # Fourth activation function
            nn.Linear(in_d, in_d),  # Fifth layer
            nn.ReLU(),             # Fifth activation function
            nn.Linear(in_d, in_d)      # Output layer
            # No activation here, apply softmax or sigmoid externally if needed, depending on your loss function
        )

    def forward(self, x):
        return self.layers(x)
    
class LinearClassifier(nn.Module):
    def __init__(self, num_classes: int, in_d: int = 1536, activation_needed: bool = True):
        super(LinearClassifier, self).__init__()
        if activation_needed:
            self.layers = nn.Sequential(
                nn.ReLU(),             # First activation function
                nn.Linear(in_d, num_classes), # First layer
            )
        else:
            self.layers = nn.Linear(in_d, num_classes)

    def forward(self, x):
        return self.layers(x)
    
def get_sequence_embeddings(mask, model_output):
    mask = mask.float()
    d = {k: v for k, v in torch.nonzero(mask).cpu().numpy()} # dict of sep tokens k = ab index, v = index of final position where mask = 1 
    # make sep token invisible
    for i in d:
        mask[i, d[i]] = 0
    mask[:, 0] = 0.0 # make cls token invisible
    mask = mask.unsqueeze(-1).expand(model_output.last_hidden_state.size())
    sum_embeddings = torch.sum(model_output.last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return sum_embeddings / sum_mask  # sum_mask means length of unmasked positions


class AbLangContrastive(nn.Module):
    def __init__(self, add_mixer: bool = True, use_cls: bool = False):
        super().__init__()
        self.roberta_heavy = AutoModel.from_pretrained(ablang_hc_hug_path, trust_remote_code=True)
        self.roberta_light = AutoModel.from_pretrained(ablang_lc_hug_path, trust_remote_code=True)
        self.config = self.roberta_heavy.config

        if add_mixer:
            self.mixer = Mixer(in_d=1536)
        else:
            self.mixer = None

        self.use_cls = use_cls

    def forward(self, h_input_ids, h_attention_mask, l_input_ids, l_attention_mask, **kwargs):
        outputs_h = self.roberta_heavy(input_ids=h_input_ids.to(torch.int64), attention_mask=h_attention_mask)
        outputs_l = self.roberta_light(input_ids=l_input_ids.to(torch.int64), attention_mask=l_attention_mask)

        if self.use_cls:
            pooled_output_h = outputs_h.last_hidden_state[:, 0, :]
            pooled_output_l = outputs_l.last_hidden_state[:, 0, :]
        else:
            pooled_output_h = get_sequence_embeddings(h_attention_mask, outputs_h)
            pooled_output_l = get_sequence_embeddings(l_attention_mask, outputs_l)

        pooled_output = torch.cat([pooled_output_h, pooled_output_l], dim=1)

        if self.mixer is not None:
            pooled_output = self.mixer(pooled_output)
            
        # Normalize embeddings for contrastive loss
        embedding = F.normalize(pooled_output, p=2, dim=1)
        return embedding


class AbLang1LinearEmbedder(nn.Module):
    def __init__(self, add_mixer: bool = True, use_cls: bool = False, num_classes: int = 16):
        super().__init__()
        self.roberta_heavy = AutoModel.from_pretrained(ablang_hc_hug_path, trust_remote_code=True)
        self.roberta_light = AutoModel.from_pretrained(ablang_lc_hug_path, trust_remote_code=True)
        self.config = self.roberta_heavy.config

        if add_mixer:
            self.mixer = Mixer(in_d=1536)
            self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(1536, num_classes)) # First layer           
        else:
            self.mixer = None
            self.classifier = nn.Linear(1536, num_classes)

        self.use_cls = use_cls

    def forward(self, h_input_ids, h_attention_mask, l_input_ids, l_attention_mask, return_embedding=False, **kwargs):
        # First do the two individual
        outputs_h = self.roberta_heavy(input_ids=h_input_ids.to(torch.int64), attention_mask=h_attention_mask)
        outputs_l = self.roberta_light(input_ids=l_input_ids.to(torch.int64), attention_mask=l_attention_mask)
        # To compare
        if self.use_cls:
            pooled_output_h = outputs_h.last_hidden_state[:, 0, :]  # CLS
            pooled_output_l = outputs_l.last_hidden_state[:, 0, :]
        else:
            pooled_output_h = get_sequence_embeddings(h_attention_mask, outputs_h) # Mean pool res encodings [8, 138, 768] -> [8, 768] (don't avg over non-AA tokens)
            pooled_output_l = get_sequence_embeddings(l_attention_mask, outputs_l)

        pooled_output = torch.cat([pooled_output_h, pooled_output_l], dim=1)  # [8, 1536]
        if self.mixer is not None:
            pooled_output = self.mixer(pooled_output)
            
        embedding = F.normalize(pooled_output, p=2, dim=1)

        logits = self.classifier(embedding)  # [batch, num_classes]
        if not return_embedding:
            return logits  # loss = F.cross_entropy(logits, labels)
        else:
            return logits, embedding
    
    def get_class(self, h_input_ids, h_attention_mask, l_input_ids, l_attention_mask):
        logits = self.forward(h_input_ids, h_attention_mask, l_input_ids, l_attention_mask)
        preds = torch.argmax(logits, dim=1) # type: ignore
        return preds
    

def save_model_loss_or_roc_and_pw(model, e: str, run_specifics: dict, metrics: list):
    metrics_df = pd.DataFrame(metrics)
    last_idx = metrics_df.index[-1]

    # "ROC-AUCS" "PW-DIFFS"

    # 3 metrics we are focusing on: loss, ROC-AUC total, mean_diff

    if metrics_df["TEST_LOSS"].idxmin() == last_idx:  # If the last run is not the best run by test loss
        model_formatted_fname = f"{run_specifics['OUTPUT_FOLDER']}/model_{run_specifics['RUN_ID']}_loss_model_epoch_%s.pt"
        if run_specifics["DELETE_OLD_MODELS"]:
            for prev_model in glob(model_formatted_fname % "*"):
                    try:
                        os.remove(prev_model)
                        print(f"Successfully deleted: {prev_model}")
                    except OSError as exc:
                        print(f"Error deleting {prev_model}: {exc}")
            else:
                print("No loss model checkpoint files found to delete.")
        ############
        model.cpu()
        torch.save(model.state_dict(), model_formatted_fname % e)

    model_formatted_fname = f"{run_specifics['OUTPUT_FOLDER']}/model_{run_specifics['RUN_ID']}_altmetric_epoch_%s.pt"
    existing_f = glob(model_formatted_fname % "*")
    try:
        if not existing_f:
            model.cpu()
            torch.save(model.state_dict(), model_formatted_fname % e)
        else:
            best_prev_epoch = int(os.path.basename(existing_f[0]).split("_")[-1].split(".")[0])
            metrics_df2 = metrics_df.set_index("EPOCH")
            prev_roc = float(metrics_df2.loc[best_prev_epoch, "ROC-AUCS"])
            prev_pw = float(metrics_df2.loc[best_prev_epoch, "PW-DIFFS"])
            cur_roc = float(metrics_df.loc[last_idx, "ROC-AUCS"])
            cur_pw= float(metrics_df.loc[last_idx, "PW-DIFFS"])

            if (cur_roc >= prev_roc) and (cur_pw >= prev_pw):
                model_formatted_fname = f"{run_specifics['OUTPUT_FOLDER']}/model_{run_specifics['RUN_ID']}_altmetric_epoch_%s.pt"
                if run_specifics["DELETE_OLD_MODELS"]:
                    for prev_model in glob(model_formatted_fname % "*"):
                            try:
                                os.remove(prev_model)
                                print(f"Successfully deleted: {prev_model}")
                            except OSError as exc:
                                print(f"Error deleting {prev_model}: {exc}")
                    else:
                        print("No altmetric model checkpoint files found to delete.")
                ##########        
                model.cpu()
                torch.save(model.state_dict(), model_formatted_fname % e)
    except Exception as ex:
        print(ex)