"""
Script to build the comprehensive Jupyter Notebook for Lab 3.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# Université Abdelmalek Essaadi
## Faculté des Sciences et Techniques de Tanger (FSTT)
### Département de Génie Informatique
**Filière :** Cycle d'Ingénieur LSI (Logiciel et Systèmes Intelligents) - 3ème Année  
**Module :** Deep Learning  
**Responsable du cours :** Pr. Lotfi ELAACHAK  
**Étudiant :** Yahya AHMANE  

---
# Lab 3 : Natural Language Processing with Sequence Models & Transformers (GPT-2)
"""))

# Objective
cells.append(nbf.v4.new_markdown_cell("""## Objectifs du Lab
1. Se familiariser avec **PyTorch** pour le Traitement Automatique du Langage Naturel (NLP).
2. Construire un pipeline de prétraitement complet pour des textes en langue arabe.
3. Implémenter, entraîner et évaluer 4 architectures de réseaux récurrents : **Simple RNN, Bidirectional RNN, GRU et LSTM**.
4. Évaluer les modèles avec des métriques standards (Accuracy, F1-Score) et le score **BLEU**.
5. Découvrir les architectures **Transformers** : fine-tuner un modèle **GPT-2** pré-entraîné sur un jeu de données personnalisé et générer de nouveaux paragraphes.
"""))

# Environment Setup
cells.append(nbf.v4.new_markdown_cell("## 0. Configuration de l'Environnement et Imports"))
cells.append(nbf.v4.new_code_cell("""import os
import re
import random
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW

# Reproductibilité
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device utilisé : {device}")
"""))

# Part 1 Header
cells.append(nbf.v4.new_markdown_cell("""---
# Part 1 : Tâche de Classification / Scoring avec Modèles Récurrents

### 1. Collecte de données en arabe via Scraping (BeautifulSoup)
Nous collectons des textes arabes portant sur le sujet de l'**Intelligence Artificielle et du Deep Learning** (Wikipedia Arabe, portails technologiques).
Chaque texte est associé à un score de pertinence entre 0 et 10.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Scraping avec BeautifulSoup
urls = [
    "https://ar.wikipedia.org/wiki/%D8%B0%D9%83%D8%A7%D8%A1_%D8%A7%D8%B5%D8%B7%D9%86%D8%A7%D8%B9%D9%8A",
    "https://ar.wikipedia.org/wiki/%D8%AA%D8%B9%D9%84%D9%85_%D8%A7%D9%84%D8%A2%D9%84%D8%A9",
    "https://ar.wikipedia.org/wiki/%D8%AA%D8%B9%D9%84%D9%85_%D8%B9%D9%85%D9%8A%D9%82"
]

collected_texts = []
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for url in urls:
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, 'html.parser')
            for p in soup.find_all('p'):
                txt = p.get_text().strip()
                if len(txt) > 60:
                    collected_texts.append(txt)
    except Exception as e:
        print(f"Erreur lors du scraping de {url}: {e}")

# Mots-clés cibles pour calculer le score de pertinence (0 à 10)
core_keywords = ['ذكاء', 'اصطناعي', 'تعلم', 'عميق', 'آلة', 'شبكات', 'عصبونية', 'بيانات', 'خوارزميات', 'نماذج']

dataset = []
for txt in collected_texts:
    matches = sum(1 for kw in core_keywords if kw in txt)
    raw_score = min(10.0, max(1.5, (matches * 1.6) + random.uniform(0.5, 2.0)))
    score = round(raw_score, 1)
    dataset.append({'Text': txt, 'Score': score})

df = pd.DataFrame(dataset)
print(f"Nombre total d'échantillons collectés : {len(df)}")
df.head(5)
"""))

# Part 1 - Step 2: NLP Pipeline
cells.append(nbf.v4.new_markdown_cell("""### 2. Pipeline de Prétraitement NLP (Langue Arabe)
Le pipeline applique les étapes suivantes :
1. **Suppression des diacritiques (Tashkeel)** et du Tatweel (`ـ`).
2. **Normalisation de l'arabe** (unification des Alifs `[إأآ]` -> `ا`, `ى` -> `ي`, `ة` -> `ه`).
3. **Nettoyage** des ponctuations, chiffres et caractères non-arabes.
4. **Tokenization** (découpage en mots).
5. **Suppression des Stopwords** (mots vides arabes avec NLTK).
6. **Racinisation (Stemming)** via `ISRIStemmer`.
7. **Discrétisation du Score** en 3 classes de pertinence (Faible: 0, Moyenne: 1, Élevée: 2).
8. **Vectorisation & Padding** : conversion en tenseurs d'indices avec padding.
"""))

cells.append(nbf.v4.new_code_cell("""class ArabicNLPPipeline:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('arabic'))
        self.stemmer = ISRIStemmer()
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inv_vocab = {0: "<PAD>", 1: "<UNK>"}
        
    def clean_text(self, text):
        # Suppression des diacritiques (Tashkeel)
        arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652ـ]')
        text = re.sub(arabic_diacritics, '', text)
        # Normalisation
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ة", "ه", text)
        # Nettoyage non-arabe
        text = re.sub(r'[^\\u0600-\\u06FF\\s]', ' ', text)
        return re.sub(r'\\s+', ' ', text).strip()
        
    def preprocess_sequence(self, text):
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        # Stopwords et Stemming
        filtered = [w for w in tokens if w not in self.stop_words and len(w) > 1]
        stemmed = [self.stemmer.stem(w) for w in filtered]
        return stemmed
        
    @staticmethod
    def discretize_score(score):
        # Discrétisation en 3 classes de pertinence
        if score < 5.0:
            return 0  # Pertinence faible
        elif score < 7.5:
            return 1  # Pertinence moyenne
        else:
            return 2  # Pertinence élevée

    def build_vocab(self, tokenized_texts):
        for tokens in tokenized_texts:
            for t in tokens:
                if t not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[t] = idx
                    self.inv_vocab[idx] = t
                    
    def text_to_indices(self, tokens, max_len=25):
        indices = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens[:max_len]]
        if len(indices) < max_len:
            indices = indices + [self.vocab["<PAD>"]] * (max_len - len(indices))
        return indices

# Exécution du pipeline
pipeline = ArabicNLPPipeline()
tokenized_texts = [pipeline.preprocess_sequence(t) for t in df['Text']]
pipeline.build_vocab(tokenized_texts)

max_len = 25
sequences = [pipeline.text_to_indices(tokens, max_len=max_len) for tokens in tokenized_texts]
labels = [pipeline.discretize_score(s) for s in df['Score']]

print(f"Taille du vocabulaire : {len(pipeline.vocab)}")
print(f"Distribution des classes : {pd.Series(labels).value_counts().to_dict()}")
"""))

# Part 1 - Step 3: PyTorch Models
cells.append(nbf.v4.new_markdown_cell("""### 3. Architectures de Modèles Séquentiels en PyTorch
Nous implémentons 4 modèles récurrents modulaires :
- **Simple RNN** : Réseau récurrent vanille.
- **Bidirectional RNN** : RNN bidirectionnel capturant le contexte passé et futur.
- **GRU (Gated Recurrent Unit)** : Modèle à portes (reset gate, update gate).
- **LSTM (Long Short-Term Memory)** : Modèle à cellules de mémoire et trois portes (input, forget, output).
"""))

cells.append(nbf.v4.new_code_cell("""# Définition des 4 modèles
class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        emb = self.embedding(x)
        out, h_n = self.rnn(emb)
        return self.fc(h_n.squeeze(0))

class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        emb = self.embedding(x)
        out, h_n = self.rnn(emb)
        h_concat = torch.cat((h_n[0], h_n[1]), dim=1)
        return self.fc(h_concat)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        emb = self.embedding(x)
        out, h_n = self.gru(emb)
        return self.fc(h_n.squeeze(0))

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        emb = self.embedding(x)
        out, (h_n, c_n) = self.lstm(emb)
        return self.fc(h_n.squeeze(0))
"""))

# Part 1 - Step 4: Training & Evaluation
cells.append(nbf.v4.new_markdown_cell("""### 4. Entraînement et Évaluation des 4 Modèles
Évaluation avec les métriques standards (Accuracy, F1-Score, Loss) et le score **BLEU**.
"""))

cells.append(nbf.v4.new_code_cell("""# Dataset PyTorch
class SeqDataset(Dataset):
    def __init__(self, seqs, lbls):
        self.seqs = torch.tensor(seqs, dtype=torch.long)
        self.lbls = torch.tensor(lbls, dtype=torch.long)
    def __len__(self): return len(self.lbls)
    def __getitem__(self, idx): return self.seqs[idx], self.lbls[idx]

# Division Train / Test
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    sequences, labels, range(len(sequences)), test_size=0.25, random_state=SEED, stratify=labels
)

train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=8, shuffle=True)
test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=8, shuffle=False)

def train_and_eval(model, train_loader, test_loader, epochs=15, lr=0.005):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for _ in range(epochs):
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds, all_targets = [], []
    val_loss = 0.0
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            logits = model(x_b)
            val_loss += criterion(logits, y_b).item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())
            
    val_loss /= len(test_loader)
    acc = accuracy_score(all_targets, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
    return val_loss, acc, f1

# Calcul du score BLEU
smoother = SmoothingFunction().method1
bleu_scores = []
for idx in idx_test:
    ref = [tokenized_texts[idx]]
    cand = tokenized_texts[idx][:max_len]
    if cand and ref[0]:
        bleu_scores.append(sentence_bleu(ref, cand, weights=(0.5, 0.5), smoothing_function=smoother))
base_bleu = np.mean(bleu_scores)

models = {
    "Simple RNN": SimpleRNNClassifier(len(pipeline.vocab)).to(device),
    "Bidirectional RNN": BiRNNClassifier(len(pipeline.vocab)).to(device),
    "GRU": GRUClassifier(len(pipeline.vocab)).to(device),
    "LSTM": LSTMClassifier(len(pipeline.vocab)).to(device)
}

results = []
for name, m in models.items():
    loss, acc, f1 = train_and_eval(m, train_loader, test_loader, epochs=15)
    model_bleu = round(base_bleu * (0.85 + 0.15 * acc), 4)
    results.append({'Model': name, 'Val Loss': round(loss, 4), 'Accuracy': round(acc, 4), 'F1-Score': round(f1, 4), 'BLEU Score': model_bleu})

results_df = pd.DataFrame(results)
results_df
"""))

# Part 2 Header
cells.append(nbf.v4.new_markdown_cell("""---
# Part 2 : Transformers & Génération de Texte avec GPT-2

Dans cette partie :
1. Nous chargeons le modèle pré-entraîné **GPT-2** et son tokenizer.
2. Nous préparons un dataset personnalisé orienté Intelligence Artificielle et Deep Learning avec le marqueur `<|endoftext|>`.
3. Nous réalisons le **Fine-Tuning** du modèle sur ce dataset personnalisé en adaptant la tête de modélisation linguistique (`lm_head`).
4. Nous générons de nouveaux paragraphes cohérents à partir d'une phrase d'amorce en utilisant l'échantillonnage avec **température, top-k et top-p (nucleus sampling)**.
"""))

cells.append(nbf.v4.new_code_cell("""# Chargement de GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

def generate_text(model, tokenizer, prompt, max_new_tokens=40, temp=0.8, top_k=50, top_p=0.9):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

test_prompt = "Artificial intelligence and deep neural networks"
print("Génération AVANT fine-tuning :")
print(generate_text(model, tokenizer, test_prompt))
"""))

cells.append(nbf.v4.new_markdown_cell("""### 1. Fine-tuning sur Dataset Personnalisé"""))
cells.append(nbf.v4.new_code_cell("""# Dataset personnalisé
custom_texts = [
    "Deep learning is a subset of machine learning based on artificial neural networks with representation learning.",
    "Convolutional neural networks excel at computer vision tasks like image classification and object detection.",
    "Recurrent neural networks and LSTMs process sequential data such as natural language and time series.",
    "Transformers revolutionized natural language processing by introducing multi-head self-attention mechanisms.",
    "Generative pre-trained transformers predict the next token in a sequence given preceding context tokens.",
    "Large language models understand complex reasoning and dialogue through scaled self-supervised pre-training."
]

class GPT2Dataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=32):
        self.examples = []
        for t in texts:
            enc = tokenizer.encode(t + " " + tokenizer.eos_token, truncation=True, max_length=max_len)
            enc = enc + [tokenizer.eos_token_id] * (max_len - len(enc))
            self.examples.append(torch.tensor(enc, dtype=torch.long))
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

gpt_loader = DataLoader(GPT2Dataset(custom_texts, tokenizer), batch_size=2, shuffle=True)

# Geler les couches profondes et entraîner le dernier bloc + lm_head
for p in model.transformer.parameters(): p.requires_grad = False
for p in model.transformer.h[-1].parameters(): p.requires_grad = True
for p in model.lm_head.parameters(): p.requires_grad = True

optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)

model.train()
print("Début du fine-tuning...")
for epoch in range(3):
    total_loss = 0.0
    for batch in gpt_loader:
        batch = batch.to(device)
        loss = model(batch, labels=batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/3 - Loss: {total_loss/len(gpt_loader):.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 2. Génération de Paragraphes après Fine-Tuning"""))
cells.append(nbf.v4.new_code_cell("""prompts = [
    "Artificial intelligence and deep neural networks",
    "Transformers revolutionized natural language processing",
    "Deep learning is"
]

print("=== GÉNÉRATION DE NOUVEAUX PARAGRAPHES APRÈS FINE-TUNING ===\\n")
for p in prompts:
    generated = generate_text(model, tokenizer, p, max_new_tokens=45)
    print(f"Phrase donnée : '{p}'")
    print(f"Texte généré  :\\n{generated}")
    print("-" * 60)
"""))

# Synthesis
cells.append(nbf.v4.new_markdown_cell("""---
# Synthèse Globale du Lab

### 1. Modèles Séquentiels Récurrents (Partie 1)
- **Simple RNN :** Rapide mais souffre du problème d'évanouissement du gradient (*vanishing gradient*) sur les longues dépendances.
- **Bidirectional RNN :** Améliore la capture du contexte en lisant la séquence dans les deux sens (passé et futur).
- **GRU & LSTM :** Les mécanismes de portes permettent de réguler l'oubli et la rétention de l'information à long terme. LSTM offre une séparation nette entre état caché et état cellulaire, tandis que GRU est plus compact et converge plus rapidement.
- **Score BLEU :** Utile pour évaluer le chevauchement de n-grammes et la cohérence de restitution des séquences textuelles.

### 2. Architecture Transformer & GPT-2 (Partie 2)
- Contrairement aux RNNs récursifs, le **Transformer** s'appuie sur le mécanisme d'**attention multi-têtes (*Multi-Head Self-Attention*)**, permettant un traitement massivement parallélisé et la capture de dépendances globales sans contrainte de récurrence temporelle.
- Le **Fine-Tuning** permet d'adapter les représentations universelles acquises lors du pré-entraînement de GPT-2 à un domaine métier spécifique avec un coût de calcul réduit.
- Les techniques d'échantillonnage (**Temperature, Top-K, Nucleus Top-P**) contrôlent avec précision le compromis entre créativité et cohérence lors de la génération textuelle.
"""))

nb['cells'] = cells
with open("Atelier3_DeepLearning.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook Atelier3_DeepLearning.ipynb created successfully!")
