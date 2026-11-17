"""
Part 2: Transformers & Text Generation with Pretrained GPT-2
Lab 3 - Deep Learning
Student: Yahya Ahmane (LSI 3 - FST Tanger)
Reference Tutorial: https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# 1. Custom Dataset Preparation
# =====================================================================
def get_custom_dataset_text():
    """
    Creates a customized dataset centered on Deep Learning, AI concepts, and scientific insights.
    Appends the standard GPT-2 delimiter <|endoftext|> to each entry as demonstrated in the tutorial.
    """
    entries = [
        "Deep learning is a subset of machine learning based on artificial neural networks with representation learning.",
        "Convolutional neural networks excel at computer vision tasks like image classification, object detection, and segmentation.",
        "Recurrent neural networks and LSTMs process sequential data such as natural language and time series forecasting.",
        "Transformers revolutionized natural language processing by introducing multi-head self-attention mechanisms without recurrence.",
        "Generative pre-trained transformers predict the next token in a sequence given preceding context tokens.",
        "Transfer learning allows deep learning models trained on massive datasets to be adapted to specific downstream tasks with minimal data.",
        "Backpropagation computes the gradient of the loss function with respect to each model weight using the chain rule of calculus.",
        "Optimization algorithms like Adam and AdamW adapt the learning rate during training for faster and more stable convergence.",
        "Large language models understand complex reasoning, programming syntax, and conversational dialogue through scaled self-supervised training.",
        "Reinforcement learning from human feedback aligns neural network outputs with human intentions, safety guidelines, and factual accuracy.",
        "Attention mechanisms compute similarity scores between query, key, and value vectors to dynamically focus on relevant parts of the input.",
        "Artificial intelligence will transform healthcare by accelerating drug discovery and assisting medical diagnosis through computer vision."
    ]
    
    os.makedirs("data", exist_ok=True)
    custom_dataset_path = "data/custom_dataset.txt"
    with open(custom_dataset_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + " <|endoftext|>\n")
            
    print(f"Custom dataset created at {custom_dataset_path} with {len(entries)} items.")
    return entries

class GPT2CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=32):
        self.examples = []
        for text in texts:
            encoded = tokenizer.encode(text + " " + tokenizer.eos_token, truncation=True, max_length=max_len)
            if len(encoded) < max_len:
                encoded = encoded + [tokenizer.eos_token_id] * (max_len - len(encoded))
            self.examples.append(torch.tensor(encoded, dtype=torch.long))
            
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]

# =====================================================================
# 2. Text Generation Helper (Top-k / Nucleus Sampling)
# =====================================================================
def generate_paragraph(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8, top_k=50, top_p=0.9):
    """
    Generates text continuation given a prompt using sampling with temperature and top-k filtering.
    Follows the sampling principles in the tutorial gist.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# =====================================================================
# 3. Fine-Tuning Function (Lightweight & Fast Transfer Learning)
# =====================================================================
def fine_tune_gpt2(model, tokenizer, dataset, epochs=3, batch_size=2, lr=5e-5):
    """
    Fine-tunes GPT-2 model on custom dataset using PyTorch DataLoader and AdamW.
    Fine-tunes the top transformer block and language modeling head.
    """
    # Freeze bottom transformer blocks for fast CPU fine-tuning
    for param in model.transformer.parameters():
        param.requires_grad = False
    # Unfreeze last block and lm_head
    for param in model.transformer.h[-1].parameters():
        param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr)
    
    model.train()
    print(f"\nStarting fine-tuning GPT-2 for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")
        
    print("Fine-tuning completed successfully!")

# =====================================================================
# Main Execution for Part 2
# =====================================================================
def run_part2():
    print("=" * 60)
    print("LAB 3 - PART 2: TRANSFORMER & TEXT GENERATION (GPT-2)")
    print("=" * 60)
    
    # 1. Load Pre-trained GPT-2
    print("\n[1/3] Loading Pretrained GPT-2 Model and Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    
    prompt = "Artificial intelligence and deep neural networks"
    print(f"\n[2/3] Generating text BEFORE fine-tuning for prompt:")
    print(f"Prompt: \"{prompt}\"")
    text_before = generate_paragraph(model, tokenizer, prompt, max_new_tokens=40)
    print(f"\nResult (Pretrained Base Model):\n{text_before}\n")
    
    # 2. Prepare custom dataset
    raw_texts = get_custom_dataset_text()
    dataset = GPT2CustomDataset(raw_texts, tokenizer, max_len=32)
    
    # 3. Fine-tune model
    print("\n[3/3] Fine-tuning GPT-2 on Customized Dataset...")
    fine_tune_gpt2(model, tokenizer, dataset, epochs=3, batch_size=2, lr=5e-5)
    
    print("\n" + "=" * 60)
    print("GENERATING NEW PARAGRAPHS ACCORDING TO GIVEN SENTENCES")
    print("=" * 60)
    
    test_prompts = [
        "Artificial intelligence and deep neural networks",
        "Transformers revolutionized natural language processing",
        "Deep learning is"
    ]
    
    for idx, p in enumerate(test_prompts, 1):
        gen_text = generate_paragraph(model, tokenizer, p, max_new_tokens=45)
        print(f"\n[Generation Sample {idx}]")
        print(f"Given Sentence : \"{p}\"")
        print(f"Generated Text :\n{gen_text}")
        print("-" * 50)

if __name__ == "__main__":
    run_part2()
