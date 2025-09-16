# AbLangRBD1: Contrastive-Learned Antibody Embeddings for SARS-CoV-2 RBD Binding

🦠 **Epitope-aware antibody embeddings for SARS-CoV-2 RBD therapeutic discovery**

[![Paper](https://img.shields.io/badge/Paper-bioRxiv-red)](https://doi.org/10.1101/2025.02.25.640114)
[![Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-blue)](https://huggingface.co/clint-holt/AbLangRBD1)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **AbLangRBD1** generates 1536-dimensional embeddings where SARS-CoV-2 RBD antibodies targeting similar epitopes cluster together - enabling rapid epitope classification, therapeutic discovery, and vaccine analysis.

## 🔬 Model Description

**AbLangRBD1** is a fine-tuned antibody language model specifically designed for SARS-CoV-2 RBD-binding antibodies. Using contrastive learning on paired heavy and light chain sequences, the model learns epitope-specific representations that enable:

- **🎯 RBD Epitope Classification**: Compare antibodies against reference databases
- **🔍 Therapeutic Discovery**: Find antibodies with similar epitope specificity
- **🦠 Vaccine Analysis**: Analyze repertoire shifts after RBD vaccination
- **📊 Cross-reactivity Studies**: Identify broadly neutralizing antibodies

### Architecture

```
Heavy Chain Seq → [AbLang Heavy] → 768-dim → |
                                              | → [Concatenate] → [Mixer Network] → 1536-dim Paired Embedding
Light Chain Seq → [AbLang Light] → 768-dim → |
```

## 🚀 Quick Start

### 📥 Model Access

**🤗 All model components are hosted on HuggingFace Hub:**

🔗 **[clint-holt/AbLangRBD1](https://huggingface.co/clint-holt/AbLangRBD1)**

```bash
# Clone this repository for inference code
git clone https://github.com/Clint-Holt/AbLangRBD1.git
cd AbLangRBD1

# Install dependencies
pip install torch pandas transformers safetensors huggingface_hub

# Run inference examples
cd Inference
python quick_start_example.py
```

### ⚡ Basic Usage

```python
import torch
from transformers import AutoTokenizer
from Inference.ablangpaired_model import AbLangPaired, AbLangPairedConfig
from huggingface_hub import hf_hub_download

# Load model from HuggingFace Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = hf_hub_download(repo_id="clint-holt/AbLangRBD1", filename="model.safetensors")

config = AbLangPairedConfig(checkpoint_filename=model_path)
model = AbLangPaired(config, device).eval()

# Load tokenizers from HuggingFace Hub
heavy_tokenizer = AutoTokenizer.from_pretrained("clint-holt/AbLangRBD1", subfolder="heavy_tokenizer")
light_tokenizer = AutoTokenizer.from_pretrained("clint-holt/AbLangRBD1", subfolder="light_tokenizer")

# Your RBD antibody sequences
heavy_chain = "EVQLVESGGGFVQPGRSLRLSCAASGFIMDDYAMHWVRQAPGKGLEWVSGISWNSGTRGYADSVKGRFTVSRDNAKNSFYLQMNSLRAADTAVYYCAKDHGPWIAANGHYFDYWGQGTLVTVSS"
light_chain = "QSVLTQPPSASGTPGQRVTISCSGSKSNIGSNPVNWYQQLPGTAPKLLIYSNNERPSGVPARFSGSKSGTSASLAISGLQSEDEADYYCVTWDDSLNGWVFGGGTKLTVL"

# Generate embedding
h_tokens = heavy_tokenizer(" ".join(heavy_chain), return_tensors="pt")
l_tokens = light_tokenizer(" ".join(light_chain), return_tensors="pt")

with torch.no_grad():
    embedding = model(
        h_input_ids=h_tokens['input_ids'].to(device),
        h_attention_mask=h_tokens['attention_mask'].to(device),
        l_input_ids=l_tokens['input_ids'].to(device),
        l_attention_mask=l_tokens['attention_mask'].to(device)
    )

print(f"Generated embedding shape: {embedding.shape}")  # (1, 1536)
```

## 📁 Repository Structure

```
AbLangRBD1/
├── Inference/                          # 🚀 Main inference code
│   ├── ablangpaired_model.py          # Core model implementation
│   ├── quick_start_example.py         # Command-line example
│   ├── rbd_inference_examples.ipynb   # Comprehensive Jupyter examples
│   ├── requirements.txt               # Python dependencies
│   └── README.md                      # Inference documentation
├── README.md                          # This file
├── LICENSE                           # MIT license
└── CLAUDE.md                         # Development notes
```

## 📊 Training Data

- **Source**: 3,195 SARS-CoV-2 RBD-binding antibodies from deep mutational scanning studies
- **References**: [Cao et al. 2023](https://doi.org/10.1038/s41586-022-05644-7), [Cao et al. 2022](https://doi.org/10.1038/s41586-022-04980-y)
- **Selection**: 3,093 antibodies with confirmed binding to SARS-CoV-2 index strain
- **Data Splits**: Clone-group aware splitting (80% train, 10% validation, 10% test)

## 💡 Use Cases

### 1. **🦠 RBD Epitope Classification**
Compare antibodies with unknown epitopes against reference databases to predict epitope class.

### 2. **🔍 Therapeutic Discovery**
Search large antibody databases to find candidates targeting specific RBD epitopes.

### 3. **📊 Vaccine Analysis**
Analyze B cell repertoire shifts following RBD vaccination by comparing pre/post vaccination samples.

### 4. **🧬 Cross-reactivity Prediction**
Identify antibodies likely to cross-react with SARS-CoV-2 variants or related coronaviruses.

## 🔗 Resources

- **🤗 HuggingFace Model**: [clint-holt/AbLangRBD1](https://huggingface.co/clint-holt/AbLangRBD1)
- **📄 Paper**: [bioRxiv](https://doi.org/10.1101/2025.02.25.640114)
- **💻 Inference Code**: [`Inference/`](Inference/) directory
- **📓 Examples**: [`Inference/rbd_inference_examples.ipynb`](Inference/rbd_inference_examples.ipynb)

## 📚 Citation

If you use AbLangRBD1 in your research, please cite our paper:

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

## 🏛️ Institution

**Vanderbilt Center for Antibody Therapeutics**
Vanderbilt University Medical Center, Nashville, TN 37232, USA

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
