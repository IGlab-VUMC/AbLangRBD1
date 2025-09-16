#!/usr/bin/env python3
"""
AbLangRBD1 Quick Start Example

This script demonstrates how to use AbLangRBD1 to generate embeddings for SARS-CoV-2 RBD-binding antibody sequences.
Run this script after downloading the model weights to verify your installation.

Usage:
    python quick_start_example.py

Requirements:
    - torch
    - pandas
    - transformers
    - safetensors
    - Model weights from HuggingFace Hub
"""

import torch
import pandas as pd
from transformers import AutoTokenizer
from ablangpaired_model import AbLangPaired, AbLangPairedConfig
import os
import sys

def main():
    print("AbLangRBD1 Quick Start Example")
    print("SARS-CoV-2 RBD Antibody Embedding Model")
    print("=" * 60)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model from HuggingFace Hub
    print("\nLoading model from HuggingFace Hub...")
    try:
        from huggingface_hub import hf_hub_download

        # Download model weights if not present
        model_path = "model.safetensors"
        if not os.path.exists(model_path):
            print("📥 Downloading model weights from HuggingFace Hub...")
            model_path = hf_hub_download(
                repo_id="clint-holt/AbLangRBD1",
                filename="model.safetensors",
                cache_dir="."
            )

        config = AbLangPairedConfig(checkpoint_filename=model_path)
        model = AbLangPaired(config, device).to(device)
        model.eval()
        print("✅ Model loaded successfully!")

    except ImportError:
        print("❌ huggingface_hub not installed. Please install with:")
        print("   pip install huggingface_hub")
        print("\nOr download manually:")
        print('   curl -L "https://huggingface.co/clint-holt/AbLangRBD1/resolve/main/model.safetensors?download=true" -o model.safetensors')
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Load tokenizers from HuggingFace Hub
    print("\nLoading tokenizers from HuggingFace Hub...")
    try:
        heavy_tokenizer = AutoTokenizer.from_pretrained("clint-holt/AbLangRBD1", subfolder="heavy_tokenizer")
        light_tokenizer = AutoTokenizer.from_pretrained("clint-holt/AbLangRBD1", subfolder="light_tokenizer")
        print("✅ Tokenizers loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading tokenizers: {e}")
        print("📡 This requires internet connection to download from HuggingFace")
        sys.exit(1)

    # Example SARS-CoV-2 RBD antibody sequences
    print("\n🦠 Processing example SARS-CoV-2 RBD antibody sequences...")

    # Example RBD-binding antibody from training data
    heavy_chain = "EVQLVESGGGFVQPGRSLRLSCAASGFIMDDYAMHWVRQAPGKGLEWVSGISWNSGTRGYADSVKGRFTVSRDNAKNSFYLQMNSLRAADTAVYYCAKDHGPWIAANGHYFDYWGQGTLVTVSS"
    light_chain = "QSVLTQPPSASGTPGQRVTISCSGSKSNIGSNPVNWYQQLPGTAPKLLIYSNNERPSGVPARFSGSKSGTSASLAISGLQSEDEADYYCVTWDDSLNGWVFGGGTKLTVL"

    print(f"Heavy chain: {heavy_chain[:50]}...")
    print(f"Light chain: {light_chain[:50]}...")

    # Tokenize sequences (add spaces between amino acids)
    print("\nTokenizing sequences...")
    h_tokens = heavy_tokenizer(" ".join(heavy_chain), return_tensors="pt")
    l_tokens = light_tokenizer(" ".join(light_chain), return_tensors="pt")

    # Generate embedding
    print("\nGenerating RBD epitope-aware embedding...")
    try:
        with torch.no_grad():
            embedding = model(
                h_input_ids=h_tokens['input_ids'].to(device),
                h_attention_mask=h_tokens['attention_mask'].to(device),
                l_input_ids=l_tokens['input_ids'].to(device),
                l_attention_mask=l_tokens['attention_mask'].to(device)
            )

        print(f"✅ Generated embedding shape: {embedding.shape}")
        print(f"Embedding dimensionality: {embedding.shape[1]}")
        print(f"First 5 embedding values: {embedding[0][:5].tolist()}")

    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        sys.exit(1)

    # Demonstrate batch processing with multiple RBD antibodies
    print("\nDemonstrating batch processing with multiple RBD antibodies...")

    # Create a dataset with multiple SARS-CoV-2 RBD antibodies
    example_data = {
        'HC_AA': [
            heavy_chain,  # Example 1 - original
            "QVQLQESGPGLVKPSETLSLTCTVSGGSISSSYYWTWIRQPPGKGLEWIGSIYHSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARGHWLVGGFDYWGQGTLVTVSS",  # Example 2 - different epitope
            "EVQLVESGGGLVQPGGSLRLSCAASGFTFRDYAMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARDHGGDHDYWGQGTLVTVSS"  # Example 3 - different epitope
        ],
        'LC_AA': [
            light_chain,  # Example 1 - original
            "DIQMTQSPSSLSASVGDRVTITCKASQDIYSSLSWYQQKPGKAPKLLIYSTSRLNSGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCQHYYSAPMTFGQGTKLEIK",  # Example 2 - different epitope
            "QSVLTQPPSVSAAPGQKVTISCSGSSSNIGSNYVSWYQQLPGTAPKLVIYSNNQRPSGVPDRFSGSKSGTSASLAISGLQSEDEADYYCATWDDSLSGYVFGTGTKVTVL"  # Example 3 - different epitope
        ]
    }

    df = pd.DataFrame(example_data)

    # Preprocess sequences
    df["PREPARED_HC_SEQ"] = df["HC_AA"].apply(lambda x: " ".join(list(x)))
    df["PREPARED_LC_SEQ"] = df["LC_AA"].apply(lambda x: " ".join(list(x)))

    # Tokenize batch
    h_tokens_batch = heavy_tokenizer(df["PREPARED_HC_SEQ"].tolist(), padding='longest', return_tensors="pt")
    l_tokens_batch = light_tokenizer(df["PREPARED_LC_SEQ"].tolist(), padding='longest', return_tensors="pt")

    # Generate embeddings for batch
    with torch.no_grad():
        embeddings_batch = model(
            h_input_ids=h_tokens_batch['input_ids'].to(device),
            h_attention_mask=h_tokens_batch['attention_mask'].to(device),
            l_input_ids=l_tokens_batch['input_ids'].to(device),
            l_attention_mask=l_tokens_batch['attention_mask'].to(device)
        )

    print(f"Generated batch embeddings shape: {embeddings_batch.shape}")
    print(f"Number of RBD antibodies processed: {embeddings_batch.shape[0]}")

    # Calculate pairwise similarities for epitope comparison
    print("\nCalculating pairwise similarities for epitope analysis...")
    similarities = torch.cosine_similarity(embeddings_batch.unsqueeze(1), embeddings_batch.unsqueeze(0), dim=2)

    print("RBD Epitope Similarity Matrix:")
    print("(Higher values = more similar epitopes)")
    for i in range(similarities.shape[0]):
        row = " ".join([f"{similarities[i][j]:.3f}" for j in range(similarities.shape[1])])
        print(f"   Ab{i+1}: [{row}]")

    print("\n🎉 AbLangRBD1 quick start example completed successfully!")


if __name__ == "__main__":
    main()