import data_handling
import models
import torch

def get_run_specifics(run_key: str) -> dict:  
    run_specifics = {  # balanced_sampling(df: pd.DataFrame, train_fract: float, epitope_col: str, group_col: str = '')
        
        'SimCLR2_250129f': {
            # General items
            'OUTPUT_FOLDER': '/dors/iglab/Members/holtcm/defining_pub_clone/ml_results/RBD/contrastive_classifiers/SimCLR-2-nopre_20250129f',
            'NUM_EPOCHS': 400, ############ 125
            'RUN_ID': 'BATCH512_12EP',
            'DELETE_OLD_MODELS': True,
            'TEMPERATURE': 0.5,

            # data_handling.py items
            'DATA_FNAME': '/home/clint/defining_pub_clone/dms_data/mlprepping_ht-dms1_2_240116_c.xlsx',
            'EPS_TO_KEEP': ['A', 'B', 'C', 'D1', 'D2', 'E1', 'E2.1', 'E2.2', 'E3', 'F1', 'F2', 'F3'],
            'HELD_OUT_EPITOPES': ['A-BA1', 'B-BA1', 'D-BA1', 'F3-BA1'],
            'USE_4CLASSES': False,
            'SAMPLING_METHOD': data_handling.balanced_sampling,
            'EPITOPE_COL': "EPITOPE",
            'GROUP_COL': 'CLONOTYPE', 
            'TRAIN_FRACT': 0.8,
            'TRAIN_BATCH_SIZE': 256,  ###################
            'TEST_BATCH_SIZE': 256,
            
            # models.py specifics
            'STARTING_MODEL_FILE': '',
            'MIXER': True,
            'USE_CLS': False,
            'USE_LORA': True,
            'MODEL_CLASS': models.AbLang1LinearEmbedder,
            'TARGET_MODULES': ["query", "value"],
            'NUM_CLASSES': 12,
            'EMBEDDING_SIZE': 1536,
            'LORA_R': 16,
            'LORA_ALPHA': 32,
            'LORA_DROPOUT': 0.3,
            'OPTIMIZER': torch.optim.AdamW,
            'LEARNING_RATE': 1e-5, 
            'SCHEDULER': None,
            'MODEL_SAVE_METHOD': models.save_model_loss_or_roc_and_pw,
            
            # analysis.py items
            'PERPLEXITY': 30,
            },

        'SimCLR1_250129c': {
            # General items
            'OUTPUT_FOLDER': '/dors/iglab/Members/holtcm/defining_pub_clone/ml_results/RBD/contrastive_classifiers/SimCLR-1_202529c',
            'NUM_EPOCHS': 400, ############ 125
            'RUN_ID': 'BATCH512_12EP',
            'DELETE_OLD_MODELS': True,
            'TEMPERATURE': 0.5,

            # data_handling.py items
            'DATA_FNAME': '/home/clint/defining_pub_clone/dms_data/mlprepping_ht-dms1_2_240116_c.xlsx',
            'EPS_TO_KEEP': ['A', 'B', 'C', 'D1', 'D2', 'E1', 'E2.1', 'E2.2', 'E3', 'F1', 'F2', 'F3'],
            'HELD_OUT_EPITOPES': ['A-BA1', 'B-BA1', 'D-BA1', 'F3-BA1'],
            'USE_4CLASSES': False,
            'SAMPLING_METHOD': data_handling.balanced_sampling,
            'EPITOPE_COL': "EPITOPE",
            'GROUP_COL': 'CLONOTYPE', 
            'TRAIN_FRACT': 0.8,
            'TRAIN_BATCH_SIZE': 256,  ###################
            'TEST_BATCH_SIZE': 256,
            
            # models.py specifics
            'STARTING_MODEL_FILE': 'model_batchsize16-2_loss_model_epoch_093.pt',
            'MIXER': True,
            'USE_CLS': False,
            'USE_LORA': True,
            'MODEL_CLASS': models.AbLang1LinearEmbedder,
            'TARGET_MODULES': ["query", "value"],
            'NUM_CLASSES': 12,
            'EMBEDDING_SIZE': 1536,
            'LORA_R': 16,
            'LORA_ALPHA': 32,
            'LORA_DROPOUT': 0.3,
            'OPTIMIZER': torch.optim.AdamW,
            'LEARNING_RATE': 1e-5, 
            'SCHEDULER': None,
            'MODEL_SAVE_METHOD': models.save_model_loss_or_roc_and_pw,
            
            # analysis.py items
            'PERPLEXITY': 30,
            },
        
        'ablang1_250127c': {
            # General items
            'OUTPUT_FOLDER': '/dors/iglab/Members/holtcm/defining_pub_clone/ml_results/RBD/linear_classifiers/ABLANG1_16_20250127c_CONTRASTIVE_STARTING_POINT',
            'NUM_EPOCHS': 125, ############ 125
            'RUN_ID': 'BATCH8_12EP',
            'DELETE_OLD_MODELS': True,

            # data_handling.py items
            'DATA_FNAME': '/home/clint/defining_pub_clone/dms_data/mlprepping_ht-dms1_2_240116_c.xlsx',
            'EPS_TO_KEEP': ['A', 'B', 'C', 'D1', 'D2', 'E1', 'E2.1', 'E2.2', 'E3', 'F1', 'F2', 'F3'],
            'HELD_OUT_EPITOPES': ['A-BA1', 'B-BA1', 'D-BA1', 'F3-BA1'],
            'USE_4CLASSES': False,
            'SAMPLING_METHOD': data_handling.balanced_sampling,
            'EPITOPE_COL': "EPITOPE",
            'GROUP_COL': 'CLONOTYPE', 
            'TRAIN_FRACT': 0.8,
            'TRAIN_BATCH_SIZE': 8,  ###################
            'TEST_BATCH_SIZE': 128,
            
            # models.py specifics
            'MIXER': True,
            'USE_CLS': False,
            'USE_LORA': True,
            'MODEL_CLASS': models.AbLang1LinearEmbedder,
            'TARGET_MODULES': ["query", "value"],
            'NUM_CLASSES': 12,
            'EMBEDDING_SIZE': 1536,
            'LORA_R': 16,
            'LORA_ALPHA': 32,
            'LORA_DROPOUT': 0.3,
            'OPTIMIZER': torch.optim.AdamW,
            'LEARNING_RATE': 1e-5, 
            'SCHEDULER': None,
            'MODEL_SAVE_METHOD': models.save_model_loss_or_roc_and_pw,
            
            # analysis.py items
            'PERPLEXITY': 30,
            },
        
        'ablang1_250126a': {
            # General items
            'OUTPUT_FOLDER': '/mnt/hd2/clint/ml_results/linear_classifiers/ABLANG1Mixed_8or16batch_12epitopes_125epochs_20250127a/',
            'NUM_EPOCHS': 125, ############ 125
            'RUN_ID': 'BATCH8_12EP',
            'DELETE_OLD_MODELS': True,

            # data_handling.py items
            'DATA_FNAME': '/home/clint/defining_pub_clone/dms_data/mlprepping_ht-dms1_2_240116_c.xlsx',
            'EPS_TO_KEEP': ['A', 'B', 'C', 'D1', 'D2', 'E1', 'E2.1', 'E2.2', 'E3', 'F1', 'F2', 'F3'],
            'HELD_OUT_EPITOPES': ['A-BA1', 'B-BA1', 'D-BA1', 'F3-BA1'],
            'USE_4CLASSES': False,
            'SAMPLING_METHOD': data_handling.balanced_sampling,
            'EPITOPE_COL': "EPITOPE",
            'GROUP_COL': 'CLONOTYPE', 
            'TRAIN_FRACT': 0.8,
            'TRAIN_BATCH_SIZE': 8,  ###################
            'TEST_BATCH_SIZE': 128,
            
            # models.py specifics
            'MIXER': True,
            'USE_CLS': False,
            'USE_LORA': True,
            'MODEL_CLASS': models.AbLang1LinearEmbedder,
            'TARGET_MODULES': ["query", "value"],
            'NUM_CLASSES': 12,
            'EMBEDDING_SIZE': 1536,
            'LORA_R': 16,
            'LORA_ALPHA': 32,
            'LORA_DROPOUT': 0.3,
            'OPTIMIZER': torch.optim.AdamW,
            'LEARNING_RATE': 1e-5, 
            'SCHEDULER': None,
            'MODEL_SAVE_METHOD': models.save_model_loss_or_roc_and_pw,
            
            # analysis.py items
            'PERPLEXITY': 30,
            },


        
        
        'ablang1_250123b': {
            # General items
            'OUTPUT_FOLDER': '/mnt/hd2/clint/ml_results/linear_classifiers/ABLANG1Mixed_12epitopes_125epochs_varyingbatches_20250124a',
            'NUM_EPOCHS': 125,
            'RUN_ID': 'BATCH8_12EP',
            'DELETE_OLD_MODELS': True,

            # data_handling.py items
            'DATA_FNAME': '/home/clint/defining_pub_clone/dms_data/mlprepping_ht-dms1_2_240116_c.xlsx',
            'EPS_TO_KEEP': ['A', 'B', 'C', 'D1', 'D2', 'E1', 'E2.1', 'E2.2', 'E3', 'F1', 'F2', 'F3'],
            'HELD_OUT_EPITOPES': ['A-BA1', 'B-BA1', 'D-BA1', 'F3-BA1'],
            'USE_4CLASSES': True,
            'SAMPLING_METHOD': data_handling.balanced_sampling,
            'EPITOPE_COL': "EPITOPE",
            'GROUP_COL': 'CLONOTYPE', 
            'TRAIN_FRACT': 0.8,
            'TRAIN_BATCH_SIZE': 8,
            'TEST_BATCH_SIZE': 128,
            
            # models.py specifics
            'MIXER': True,
            'USE_CLS': False,
            'USE_LORA': True,
            'MODEL_CLASS': models.AbLang1LinearEmbedder,
            'TARGET_MODULES': ["query", "value"],
            'NUM_CLASSES': 12,
            'EMBEDDING_SIZE': 1536,
            'LORA_R': 16,
            'LORA_ALPHA': 32,
            'LORA_DROPOUT': 0.3,
            'OPTIMIZER': torch.optim.AdamW,
            'LEARNING_RATE': 1e-5, 
            'SCHEDULER': None,
            'MODEL_SAVE_METHOD': models.save_model_loss_or_roc_and_pw,
            
            # analysis.py items
            'PERPLEXITY': 30,
            },
        'ablang1_250123a_test': {
            # General items
            'OUTPUT_FOLDER': '/mnt/hd2/clint/ml_results/linear_classifiers/ABLANG1Mixed_4classes_125epochs_varyingbatches_20250123b',
            'NUM_EPOCHS': 125,

            # data_handling.py items
            'DATA_FNAME': '/home/clint/defining_pub_clone/dms_data/mlprepping_ht-dms1_2_240116_c.xlsx',
            'EPS_TO_KEEP': ['A', 'B', 'C', 'D1', 'D2', 'E1', 'E2.1', 'E2.2', 'E3', 'F1', 'F2', 'F3'],
            'HELD_OUT_EPITOPES': ['A-BA1', 'B-BA1', 'D-BA1', 'F3-BA1'],
            'USE_4CLASSES': True,
            'SAMPLING_METHOD': data_handling.simple_sampling, 
            'TRAIN_FRACT': 0.8,
            'TRAIN_BATCH_SIZE': 8,
            'TEST_BATCH_SIZE': 128,
            
            # models.py specifics
            'MIXER': True,
            'USE_CLS': False,
            'USE_LORA': True,
            'MODEL_CLASS': models.AbLang1LinearEmbedder,
            'TARGET_MODULES': ["query", "value"],
            'NUM_CLASSES': 4,
            'EMBEDDING_SIZE': 1536,
            'LORA_R': 16,
            'LORA_ALPHA': 32,
            'LORA_DROPOUT': 0.3,
            'OPTIMIZER': torch.optim.AdamW,
            'LEARNING_RATE': 1e-5, 
            'SCHEDULER': None,
            
            # analysis.py items
            'PERPLEXITY': 30,
            },
    }
    return run_specifics[run_key]