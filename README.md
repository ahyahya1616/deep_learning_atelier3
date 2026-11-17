# Deep Learning - Lab 3 : Natural Language Processing with Sequence Models & Transformers (GPT-2)

**Université Abdelmalek Essaadi**  
**Faculté des Sciences et Techniques de Tanger (FSTT)**  
**Département de Génie Informatique**  
**Filière :** Cycle d'Ingénieur LSI (Logiciel et Systèmes Intelligents) — 3ème Année  
**Module :** Deep Learning  
**Responsable du module :** Pr. Lotfi ELAACHAK  
**Étudiant :** Yahya AHMANE  
**Dépôt GitHub :** [https://github.com/ahyahya1616/deep_learning_atelier3](https://github.com/ahyahya1616/deep_learning_atelier3)

---

## Table des Matières
1. [Introduction et Objectifs](#introduction-et-objectifs)
2. [Structure du Projet](#structure-du-projet)
3. [Partie 1 : Tâche de Classification / Scoring (NLP & Sequence Models)](#partie-1--tâche-de-classification--scoring-nlp--sequence-models)
   - [1. Collecte de données en Arabe via Scraping](#1-collecte-de-données-en-arabe-via-scraping)
   - [2. Pipeline de Prétraitement NLP](#2-pipeline-de-prétraitement-nlp)
   - [3. Entraînement des Modèles Séquentiels (RNN, BiRNN, GRU, LSTM)](#3-entraînement-des-modèles-séquentiels-rnn-birnn-gru-lstm)
   - [4. Évaluation et Métriques (Accuracy, Loss, F1, Score BLEU)](#4-évaluation-et-métriques-accuracy-loss-f1-score-bleu)
4. [Partie 2 : Transformers & Génération de Texte (GPT-2)](#partie-2--transformers--génération-de-texte-gpt-2)
   - [1. Fine-tuning de GPT-2 sur un Dataset Personnalisé](#1-fine-tuning-de-gpt-2-sur-un-dataset-personnalisé)
   - [2. Génération de Paragraphes selon des Phrases Données](#2-génération-de-paragraphes-selon-des-phrases-données)
5. [Synthèse Globale du Lab](#synthèse-globale-du-lab)
6. [Instructions d'Exécution](#instructions-d-exécution)

---

## Introduction et Objectifs
L'objectif principal de cet atelier est de se familiariser avec **PyTorch** et l'écosystème **HuggingFace Transformers** pour construire des architectures de réseaux de neurones profonds appliquées au traitement automatique du langage naturel (NLP).

Le travail se divise en deux volets majeurs :
- **Partie 1 :** Scraping d'articles en langue arabe, mise en place d'un pipeline complet de prétraitement NLP (tokenisation, stemming, lemmatisation, stopwords, discrétisation), puis entraînement et comparaison de 4 architectures récurrentes (**Simple RNN**, **Bidirectional RNN**, **GRU**, **LSTM**) évaluées avec des métriques standards et le score **BLEU**.
- **Partie 2 :** Prise en main des modèles d'attention (Transformers) en adaptant un modèle pré-entraîné **GPT-2** via fine-tuning sur des données ciblées, puis génération de texte conditionnée par des amorces textuelles.

---

## Structure du Projet

```bash
deep_learning_atelier3/
├── data/
│   ├── arabic_dataset.csv            # Dataset scrapé en langue arabe avec scores (0 à 10)
│   └── custom_dataset.txt            # Dataset personnalisé pour le fine-tuning de GPT-2
├── part1_nlp_sequence_models.py      # Script complet Partie 1 : Scraping, NLP, RNN, BiRNN, GRU, LSTM, BLEU
├── part2_gpt2_text_generation.py     # Script complet Partie 2 : Chargement GPT-2, Fine-Tuning, Génération
├── Atelier3_DeepLearning.ipynb       # Notebook Jupyter complet prêt à exécuter (Colab / Kaggle / Local)
├── build_notebook.py                 # Script générateur du notebook
├── .gitignore                        # Exclusion des fichiers inutiles et Atelier3.pdf
└── README.md                         # Rapport complet de l'atelier
```

---

## Partie 1 : Tâche de Classification / Scoring (NLP & Sequence Models)

### 1. Collecte de données en Arabe via Scraping
À l'aide de **BeautifulSoup** et **Requests**, nous collectons des paragraphes en langue arabe traitant de l'Intelligence Artificielle et du Deep Learning (articles issus de Wikipédia Arabe et de portails scientifiques).

Chaque paragraphe est associé à un score de pertinence continu compris entre **0.0 et 10.0** calculé en fonction de la densité de mots-clés thématiques (ex. *ذكاء*, *اصطناعي*, *تعلم*, *عميق*, *شبكات عصبونية*).

Extrait du jeu de données (`data/arabic_dataset.csv`) :

| Text (Langue Arabe) | Score |
| :--- | :---: |
| الذَّكَاءُ الْاِصْطِنَاعِيُّ أو الذَّكَاءُ الصِّنَاعِيُّ هو سلوك وخصائص تتسم بها البرامج الحاسوبية محاكية للقدرات البشرية... | 6.3 |
| تشمل أبرز تطبيقات الذكاء الاصطناعي محرّكات البحث ومحركات التوصية والترجمة الآلية والسيارات ذاتية القيادة... | 8.5 |
| تشمل الأهداف التقليدية لأبحاث الذكاء الاصطناعي التفكير والتمثيل المعرفي والتخطيط والتعلم ومعالجة اللغات الطبيعية... | 10.0 |

---

### 2. Pipeline de Prétraitement NLP
La langue arabe nécessite un nettoyage rigoureux en raison de sa richesse morphologique :
1. **Suppression du Tashkeel (Diacritiques) :** Élimination des harakat (fatha, damma, kasra, sukun, tanween, shadda) et du tatweel (`ـ`).
2. **Normalisation :** Unification des différentes formes d'Alif (`أ`, `إ`, `آ` $\rightarrow$ `ا`), du Yaa final (`ى` $\rightarrow$ `ي`) et du Taa Marbouta (`ة` $\rightarrow$ `ه`).
3. **Filtrage des caractères :** Élimination de la ponctuation, des caractères latins et des chiffres pour ne conserver que les lettres arabes.
4. **Tokenisation :** Découpage du texte en unités lexicales (tokens).
5. **Suppression des Mots Vides (Stop Words) :** Utilisation de la liste de stopwords arabes de **NLTK** (701 mots outils éliminés).
6. **Racinisation (Stemming) :** Utilisation de `ISRIStemmer` de NLTK pour extraire les racines morphologiques tri-consonantiques.
7. **Discrétisation :** Transformation du score continu (0-10) en 3 classes de pertinence pour la classification :
   - **Classe 0 (Faible) :** $[0.0, 5.0[$
   - **Classe 1 (Moyenne) :** $[5.0, 7.5[$
   - **Classe 2 (Élevée) :** $[7.5, 10.0]$
8. **Vectorisation & Padding :** Construction du vocabulaire (`<PAD>`, `<UNK>`, etc.) et alignement des séquences sur une longueur fixe `max_len = 25`.

---

### 3. Entraînement des Modèles Séquentiels (RNN, BiRNN, GRU, LSTM)
Quatre architectures modulaires ont été implémentées en **PyTorch** avec une couche d'embedding partagée ($d=64$), une couche récurrente cachée ($h=64$) et une couche linéaire fully-connected de classification :

1. **Simple RNN :** Architecture standard récurrente avec fonction d'activation $\tanh$.
2. **Bidirectional RNN (BiRNN) :** Traite la séquence dans le sens avant (*forward*) et arrière (*backward*) pour capturer le contexte global passé et futur (état caché concaténé de dimension $2 \times 64 = 128$).
3. **GRU (Gated Recurrent Unit) :** Introduit deux portes (Reset Gate et Update Gate) pour contrôler le flux d'information sans état cellulaire séparé.
4. **LSTM (Long Short-Term Memory) :** Introduit un état cellulaire $c_t$ régulé par trois portes (Forget Gate, Input Gate, Output Gate) pour contrer la disparition du gradient.

Hyperparamètres :
- Optimiseur : **Adam** ($\text{lr} = 0.005$)
- Fonction de perte : **CrossEntropyLoss**
- Taille de lot (Batch Size) : 8
- Époques : 20

---

### 4. Évaluation et Métriques (Accuracy, Loss, F1, Score BLEU)

Tableau comparatif des performances obtenues sur le jeu de test :

| Modèle | Val Loss | Accuracy | F1-Score (pondéré) | Score BLEU |
| :--- | :---: | :---: | :---: | :---: |
| **Simple RNN** | 1.7910 | 0.5122 | 0.5332 | 0.4352 |
| **Bidirectional RNN** | 2.5442 | 0.4268 | 0.3939 | 0.4292 |
| **LSTM** | 2.5580 | 0.4512 | 0.4262 | 0.4309 |
| **GRU** | 3.5888 | 0.4024 | 0.4001 | 0.4275 |

**Analyse des résultats :**
- **Score BLEU :** Évalue la préservation structurelle des n-grammes de la séquence textuelle. Les scores sont stables (~0.43), confirmant une bonne conservation sémantique après prétraitement.
- Sur ce corpus ciblé, le **Simple RNN** converge rapidement en raison du nombre limité de paramètres, tandis que le **LSTM** offre une meilleure stabilité sur les représentations de phrases complexes grâce à sa cellule de mémoire.

---

## Partie 2 : Transformers & Génération de Texte (GPT-2)

### 1. Fine-tuning de GPT-2 sur un Dataset Personnalisé
En suivant le tutoriel de référence ([gist mf1024](https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7)), nous chargeons le modèle pré-entraîné **GPT-2** (124M paramètres) et son tokenizer BPE (*Byte-Pair Encoding*).

Nous préparons un dataset personnalisé axé sur le Deep Learning et l'IA en adjoignant à chaque entrée le marqueur de fin de texte `<|endoftext|>`.

**Procédure de Fine-Tuning :**
- Gel des blocs inférieurs du Transformer pour préserver les connaissances générales du modèle de base.
- Entraînement de la tête de prédiction (`lm_head`) et du dernier bloc d'attention sur les données cibles avec l'optimiseur **AdamW** ($\text{lr} = 5 \times 10^{-5}$).
- Évolution de la perte d'entraînement :
  - **Époque 1/3 :** Loss = 7.4175
  - **Époque 2/3 :** Loss = 6.7380
  - **Époque 3/3 :** Loss = 6.1652

---

### 2. Génération de Paragraphes selon des Phrases Données

Pour générer des continuations riches, cohérentes et non répétitives, nous combinons :
- **Temperature ($T = 0.8$) :** Ajuste la netteté de la distribution de probabilité sur les prochains tokens.
- **Top-K Sampling ($K = 50$) :** Restreint les candidats aux 50 tokens les plus probables.
- **Nucleus (Top-P) Sampling ($P = 0.9$) :** Restreint le choix aux tokens accumulant 90% de probabilité cumulée.
- **Pénalité de répétition :** 1.2 pour éviter les boucles textuelles.

#### Exemples de Générations Obtenues :

**1. Phrase d'amorce :** `"Artificial intelligence and deep neural networks"`  
> **Texte généré :**  
> *"Artificial intelligence and deep neural networks are still in the early stages of development. The problem is that these artificial systems don't have a lot to offer because they rely on computer programming language or software tools like python, which typically has an interface with humans for learning..."*

**2. Phrase d'amorce :** `"Transformers revolutionized natural language processing"`  
> **Texte généré :**  
> *"Transformers revolutionized natural language processing. In the process, artificial intelligence was able to adapt into human languages more quickly than ever before and create an intuitive representation of reality that is indistinguishable from real time data in a single program-driven approach with minimal effort on..."*

**3. Phrase d'amorce :** `"Deep learning is"`  
> **Texte généré :**  
> *"Deep learning is just a step away from the next big breakthrough in AI and machine-learning."*

**Comparaison Avant / Après Fine-Tuning :**  
Le modèle pré-entraîné de base produisait des phrases très génériques ou de style web brut. Après fine-tuning, le modèle génère des paragraphes nettement plus structurés, pertinents et alignés avec le vocabulaire scientifique du Deep Learning.

---

## Synthèse Globale du Lab

Au terme de cet atelier, nous retenons les enseignements fondamentaux suivants :

1. **Importance du Prétraitement pour les Langues Morphologiquement Riches (Arabe) :**
   Le nettoyage (désaccentuation du Tashkeel, normalisation de l'Alif et de la Taa Marbouta, stemming avec `ISRIStemmer`) est une étape critique pour réduire la sparsité du vocabulaire et permettre aux modèles récurrents d'apprendre efficacement.

2. **Évolution des Modèles Séquentiels Récurrents :**
   - Le **Simple RNN** est adapté aux petites séquences mais souffre du problème d'évanouissement ou d'explosion du gradient (*vanishing/exploding gradient*).
   - Le **Bidirectional RNN** fournit une vision bidirectionnelle utile pour les tâches de classification globale de texte où le contexte futur est déjà disponible.
   - Les architectures à portes (**GRU** et **LSTM**) règlent le problème d'évanouissement du gradient grâce à leurs mécanismes de filtrage et de mémorisation explicite.

3. **Supériorité des Architectures Transformers :**
   - Contrairement aux RNNs qui traitent le texte séquentiellement mot après mot, les **Transformers** s'appuient sur le mécanisme d'**Attention Multi-Têtes (*Multi-Head Self-Attention*)**, permettant un traitement parallèle et la capture de dépendances longue distance sans perte d'information temporelle.
   - Le **Transfer Learning** et le **Fine-Tuning** avec des modèles comme **GPT-2** permettent de capitaliser sur des milliards de tokens appris en amont pour accomplir des tâches génératives spécialisées avec un coût d'entraînement minimal.

---

## Instructions d'Exécution

### 1. Cloner le dépôt
```bash
git clone git@github.com:ahyahya1616/deep_learning_atelier3.git
cd deep_learning_atelier3
```

### 2. Créer un environnement virtuel et installer les dépendances
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Ou installation directe : `pip install torch transformers nltk scikit-learn pandas beautifulsoup4 requests`)*

### 3. Exécuter la Partie 1 (Scraping, NLP & Sequence Models)
```bash
python3 part1_nlp_sequence_models.py
```

### 4. Exécuter la Partie 2 (GPT-2 Fine-Tuning & Génération de Texte)
```bash
python3 part2_gpt2_text_generation.py
```

### 5. Ouvrir le Notebook interactif
Le notebook [`Atelier3_DeepLearning.ipynb`](Atelier3_DeepLearning.ipynb) peut être exécuté localement avec Jupyter Notebook / JupyterLab, ou importé directement dans **Google Colab** ou **Kaggle**.

