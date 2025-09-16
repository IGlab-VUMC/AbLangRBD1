# AbLangRBD1 Inference

**SARS-CoV-2 RBD Antibody Embedding Inference**

This directory contains inference code and examples for **AbLangRBD1**, a specialized model for generating epitope-aware embeddings of SARS-CoV-2 RBD-binding antibodies.

## About AbLangRBD1

AbLangRBD1 generates 1536-dimensional embeddings where antibodies targeting similar RBD epitopes cluster together, enabling:

- **RBD Epitope Classification**: Compare antibodies against reference databases
- **Therapeutic Discovery**: Find antibodies with similar epitope specificity
- **Vaccine Analysis**: Analyze repertoire shifts after RBD vaccination
- **Cross-reactivity Studies**: Identify broadly neutralizing antibodies

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For HuggingFace Hub integration
pip install huggingface_hub
```

### Basic Usage

```python
import torch
from transformers import AutoTokenizer
from ablangpaired_model import AbLangPaired, AbLangPairedConfig

# Load model from HuggingFace Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The model weights will be automatically downloaded from HuggingFace
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="clint-holt/AbLangRBD1", filename="model.safetensors")

config = AbLangPairedConfig(checkpoint_filename=model_path)
model = AbLangPaired(config, device).eval()

# Load tokenizers from HuggingFace Hub
heavy_tokenizer = AutoTokenizer.from_pretrained("clint-holt/AbLangRBD1", subfolder="heavy_tokenizer")
light_tokenizer = AutoTokenizer.from_pretrained("clint-holt/AbLangRBD1", subfolder="light_tokenizer")

# Your RBD antibody sequences
heavy_chain = "EVQLVESGGGFVQPGRSLRLSCAASGFIMDDYAMHWVRQAPGKGLEWVSGISWNSGTRGYADSVKGRFTVSRDNAKNSFYLQMNSLRAADTAVYYCAKDHGPWIAANGHYFDYWGQGTLVTVSS"
light_chain = "QSVLTQPPSASGTPGQRVTISCSGSKSNIGSNPVNWYQQLPGTAPKLLIYSNNERPSGVPARFSGSKSGTSASLAISGLQSEDEADYYCVTWDDSLNGWVFGGGTKLTVL"

# Tokenize (add spaces between amino acids)
h_tokens = heavy_tokenizer(" ".join(heavy_chain), return_tensors="pt")
l_tokens = light_tokenizer(" ".join(light_chain), return_tensors="pt")

# Generate embedding
with torch.no_grad():
    embedding = model(
        h_input_ids=h_tokens['input_ids'].to(device),
        h_attention_mask=h_tokens['attention_mask'].to(device),
        l_input_ids=l_tokens['input_ids'].to(device),
        l_attention_mask=l_tokens['attention_mask'].to(device)
    )

print(f"Generated embedding shape: {embedding.shape}")  # (1, 1536)
```

## 📁 Files in this Directory

| File | Description |
|------|-------------|
| `ablangpaired_model.py` | Core model implementation (AbLangPaired class) |
| `quick_start_example.py` | Simple command-line example script |
| `rbd_inference_examples.ipynb` | Comprehensive Jupyter notebook with examples |
| `config.json` | Model configuration file |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Example Applications

### 1. **RBD Epitope Classification**
```python
# Compare unknown antibody against reference database
similarities = torch.cosine_similarity(query_embedding, reference_embeddings, dim=1)
best_match_idx = torch.argmax(similarities)
predicted_epitope = reference_epitopes[best_match_idx]
```

### 2. **Batch Processing**
```python
# Process multiple antibodies efficiently
h_tokens_batch = heavy_tokenizer(heavy_sequences, padding='longest', return_tensors="pt")
l_tokens_batch = light_tokenizer(light_sequences, padding='longest', return_tensors="pt")

with torch.no_grad():
    embeddings_batch = model(
        h_input_ids=h_tokens_batch['input_ids'].to(device),
        h_attention_mask=h_tokens_batch['attention_mask'].to(device),
        l_input_ids=l_tokens_batch['input_ids'].to(device),
        l_attention_mask=l_tokens_batch['attention_mask'].to(device)
    )
```

### 3. **Epitope Similarity Analysis**
```python
# Calculate pairwise similarities
similarities = torch.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
```

## 🤗 HuggingFace Hub Integration

All model components are hosted on HuggingFace Hub for easy access:

- **🤖 Model Hub**: [clint-holt/AbLangRBD1](https://huggingface.co/clint-holt/AbLangRBD1)
- **📥 Model Weights**: Automatically downloaded via `huggingface_hub`
- **🔤 Tokenizers**: Integrated with `transformers` library

### Manual Download (if needed)
```bash
# Download model weights manually
curl -L "https://huggingface.co/clint-holt/AbLangRBD1/resolve/main/model.safetensors?download=true" -o model.safetensors
```

## 🔧 Requirements

- **Python**: ≥3.8
- **PyTorch**: ≥1.13.0
- **Transformers**: ≥4.30.0
- **Other**: pandas, safetensors, huggingface_hub

## ⚠️ Important Notes

1. **RBD-Specific**: This model is optimized for SARS-CoV-2 RBD-binding antibodies
2. **Human Antibodies**: Best performance with human antibody sequences
3. **Internet Required**: Initial setup requires internet connection for downloading from HuggingFace Hub
4. **Model Size**: Model weights are ~738MB

## Citation

If you use AbLangRBD1 in your research, please cite:

```bibtex
@article{Holt2025.02.25.640114,
    author = {Holt, Clinton M. and Janke, Alexis K. and Amlashi, Parastoo and Jamieson, Parker J. and Marinov, Toma M. and Georgiev, Ivelin S.},
    title = {Contrastive Learning Enables Epitope Overlap Predictions for Targeted Antibody Discovery},
    elocation-id = {2025.02.25.640114},
    year = {2025},
    doi = {10.1101/2025.02.25.640114},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2025/04/01/2025.02.25.640114},
    eprint = {https://www.biorxiv.org/content/early/2025/04/01/2025.02.25.640114.full.pdf},
    journal = {bioRxiv}
}
```

## 🔗 Resources

- **Paper**: [bioRxiv](https://doi.org/10.1101/2025.02.25.640114)
- **Model Hub**: [clint-holt/AbLangRBD1](https://huggingface.co/clint-holt/AbLangRBD1)
- **GitHub**: [AbLangRBD1 Repository](https://github.com/Clint-Holt/AbLangRBD1)
