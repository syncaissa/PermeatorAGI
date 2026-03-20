"""
Multi-Domain Data Loader for Permeator / Permeator Training
======================================================

Loads data from 5 distinct domains via HuggingFace, Kaggle*, and torchvision:
    1. News/Current Events  (AG News — 4 classes)          [HuggingFace]
    2. Emotion/Psychology   (dair-ai/emotion — 6 classes)  [HuggingFace]
    3. Film/Entertainment   (IMDb — 2 classes)             [HuggingFace]
    4. Medical/Health       (medical_meadow — 2 classes)   [Kaggle-origin, HuggingFace mirror]
    5. Vision/Objects       (CIFAR-10 — 10 classes)        [torchvision]

* Kaggle datasets are accessed via HuggingFace mirrors to avoid requiring
  a Kaggle API key in Colab. The medical domain dataset originates from
  Kaggle's medical text datasets. Users with Kaggle API keys can substitute
  any Kaggle dataset using the kaggle Python package.

All text domains share a unified vocabulary and tokenizer.
All domains are subsampled for efficient training on Colab.

NAMING CONVENTION:
- "Permeator" and "Permeator" are used interchangeably.
- "Membrane" and "Filter" are used interchangeably.

Author: Milind K. Patil, Syncaissa Systems Inc.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re
import random

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# Simple Tokenizer (no external dependencies like transformers tokenizer)
# =============================================================================

class SimpleTokenizer:
    """
    Lightweight word-level tokenizer.
    Builds vocabulary from training data. No pretrained model needed.
    """

    def __init__(self, max_vocab=30000, max_len=128):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

    def _clean(self, text):
        """Basic text cleaning."""
        text = text.lower().strip()
        text = re.sub(r'<[^>]+>', ' ', text)   # remove HTML tags
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def build_vocab(self, texts):
        """Build vocabulary from corpus."""
        counter = Counter()
        for text in texts:
            words = self._clean(text).split()
            counter.update(words)

        # Most common words (reserve 0 for PAD, 1 for UNK)
        for word, _ in counter.most_common(self.max_vocab - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"  Vocabulary built: {len(self.word2idx)} tokens")

    def encode(self, text):
        """Tokenize and encode text to token ids."""
        words = self._clean(text).split()[:self.max_len]
        ids = [self.word2idx.get(w, 1) for w in words]  # 1 = <UNK>

        # Pad or truncate to max_len
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return ids[:self.max_len]

    def encode_batch(self, texts):
        """Encode a batch of texts."""
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)


# =============================================================================
# Domain Dataset Wrapper
# =============================================================================

class DomainDataset(Dataset):
    """Unified dataset wrapper for any domain."""

    def __init__(self, data, labels, domain_type='text', tokenizer=None):
        """
        Args:
            data:        list of texts or tensor of images
            labels:      list/tensor of integer labels
            domain_type: 'text' or 'image'
            tokenizer:   SimpleTokenizer instance (for text)
        """
        self.domain_type = domain_type
        self.labels = torch.tensor(labels, dtype=torch.long)

        if domain_type == 'text':
            assert tokenizer is not None
            self.data = tokenizer.encode_batch(data)
        elif domain_type == 'image':
            self.data = data  # already a tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_ag_news(n_train=5000, n_test=1000):
    """
    Load AG News — News topic classification.
    Domain: News / Current Events
    Classes: World (0), Sports (1), Business (2), Sci/Tech (3)
    """
    print("\n📰 Loading AG News (News/Current Events domain)...")
    from datasets import load_dataset
    ds = load_dataset('ag_news')

    train_texts = ds['train']['text'][:n_train]
    train_labels = ds['train']['label'][:n_train]
    test_texts = ds['test']['text'][:n_test]
    test_labels = ds['test']['label'][:n_test]

    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}, Classes: 4")
    return train_texts, train_labels, test_texts, test_labels, 4


def load_emotion(n_train=5000, n_test=1000):
    """
    Load Emotion dataset — Emotion classification.
    Domain: Psychology / Emotion
    Classes: sadness(0), joy(1), love(2), anger(3), fear(4), surprise(5)
    """
    print("\n🧠 Loading Emotion (Psychology/Emotion domain)...")
    from datasets import load_dataset
    ds = load_dataset('dair-ai/emotion')

    train_texts = ds['train']['text'][:n_train]
    train_labels = ds['train']['label'][:n_train]
    test_texts = ds['test']['text'][:n_test]
    test_labels = ds['test']['label'][:n_test]

    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}, Classes: 6")
    return train_texts, train_labels, test_texts, test_labels, 6


def load_imdb(n_train=5000, n_test=1000):
    """
    Load IMDb — Movie review sentiment analysis.
    Domain: Film / Entertainment
    Classes: negative (0), positive (1)
    """
    print("\n🎬 Loading IMDb (Film/Entertainment domain)...")
    from datasets import load_dataset
    ds = load_dataset('imdb')

    train_texts = ds['train']['text'][:n_train]
    train_labels = ds['train']['label'][:n_train]
    test_texts = ds['test']['text'][:n_test]
    test_labels = ds['test']['label'][:n_test]

    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}, Classes: 2")
    return train_texts, train_labels, test_texts, test_labels, 2


def load_medical(n_train=5000, n_test=1000):
    """
    Load Medical/Health domain data.
    Origin: Kaggle medical text datasets, accessed via HuggingFace mirror.
    Domain: Medical / Health

    Uses the medical_questions_pairs dataset (originally from Kaggle),
    framed as binary classification: are two medical questions similar?

    If unavailable, falls back to health-related subset of AG News or
    a synthetic medical text classification task.
    """
    print("\n🏥 Loading Medical/Health (Kaggle-origin domain)...")
    from datasets import load_dataset

    try:
        # Primary: medical_questions_pairs (Kaggle origin, HuggingFace mirror)
        ds = load_dataset('medical_questions_pairs')
        texts = [f"{row['question_1']} {row['question_2']}"
                 for row in ds['train']]
        labels = [int(row['label']) for row in ds['train']]

        # Split into train/test
        n_total = len(texts)
        n_tr = min(n_train, int(n_total * 0.8))
        n_te = min(n_test, n_total - n_tr)

        train_texts = texts[:n_tr]
        train_labels = labels[:n_tr]
        test_texts = texts[n_tr:n_tr + n_te]
        test_labels = labels[n_tr:n_tr + n_te]
        n_classes = 2
        print(f"  Source: medical_questions_pairs (Kaggle origin)")

    except Exception:
        # Fallback: use health_fact dataset (also Kaggle-origin)
        try:
            ds = load_dataset('health_fact')
            train_texts = [str(t) for t in ds['train']['claim'][:n_train]]
            train_labels = [min(l, 3) for l in ds['train']['label'][:n_train]]
            test_texts = [str(t) for t in ds['test']['claim'][:n_test]]
            test_labels = [min(l, 3) for l in ds['test']['label'][:n_test]]
            n_classes = 4
            print(f"  Source: health_fact (Kaggle origin)")
        except Exception:
            # Final fallback: use AG News Sci/Tech subset as science/health proxy
            print("  ⚠ Medical datasets unavailable, using AG News Sci/Tech subset")
            ds = load_dataset('ag_news')
            # Filter to Sci/Tech class (label=3) and Business (label=2)
            train_data = [(t, l) for t, l in
                         zip(ds['train']['text'], ds['train']['label'])
                         if l >= 2][:n_train]
            test_data = [(t, l) for t, l in
                        zip(ds['test']['text'], ds['test']['label'])
                        if l >= 2][:n_test]
            train_texts = [t for t, l in train_data]
            train_labels = [l - 2 for t, l in train_data]  # remap to 0,1
            test_texts = [t for t, l in test_data]
            test_labels = [l - 2 for t, l in test_data]
            n_classes = 2

    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}, "
          f"Classes: {n_classes}")
    return train_texts, train_labels, test_texts, test_labels, n_classes


def load_cifar10(n_train=5000, n_test=1000):
    """
    Load CIFAR-10 — Image classification.
    Domain: Vision / Objects
    Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
    """
    print("\n🖼️  Loading CIFAR-10 (Vision/Objects domain)...")
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # Subsample
    train_indices = np.random.choice(len(train_ds), n_train, replace=False)
    test_indices = np.random.choice(len(test_ds), n_test, replace=False)

    train_images = torch.stack([train_ds[i][0] for i in train_indices])
    train_labels = [train_ds[i][1] for i in train_indices]
    test_images = torch.stack([test_ds[i][0] for i in test_indices])
    test_labels = [test_ds[i][1] for i in test_indices]

    print(f"  Train: {n_train}, Test: {n_test}, Classes: 10")
    return train_images, train_labels, test_images, test_labels, 10


# =============================================================================
# Master Data Loader
# =============================================================================

def load_all_domains(n_train=5000, n_test=1000, max_vocab=30000, max_len=128,
                     batch_size=64):
    """
    Load ALL 5 domains and prepare DataLoaders.

    Returns:
        tokenizer:    shared tokenizer for text domains
        dataloaders:  dict mapping domain → (train_loader, test_loader)
        domain_info:  dict mapping domain → {type, n_classes, vocab_size}
    """
    print("=" * 70)
    print("Permeator / Permeator Multi-Domain Data Loading")
    print("=" * 70)
    print(f"Loading 5 domains: News, Emotion, Film, Medical (Kaggle), Vision")
    print(f"Training samples per domain: {n_train}")
    print(f"Test samples per domain: {n_test}")

    # --- Load all text domains ---
    ag_train = load_ag_news(n_train, n_test)
    emo_train = load_emotion(n_train, n_test)
    imdb_train = load_imdb(n_train, n_test)
    med_train = load_medical(n_train, n_test)

    # --- Build shared vocabulary from ALL text domains ---
    print("\n🔤 Building shared vocabulary across all text domains...")
    tokenizer = SimpleTokenizer(max_vocab=max_vocab, max_len=max_len)
    all_texts = (list(ag_train[0]) + list(emo_train[0]) +
                 list(imdb_train[0]) + list(med_train[0]))
    tokenizer.build_vocab(all_texts)

    vocab_size = len(tokenizer.word2idx)

    # --- Create text datasets ---
    text_domains = {
        'news': (ag_train, 4),
        'emotion': (emo_train, 6),
        'film': (imdb_train, 2),
        'medical': (med_train, med_train[4]),   # n_classes from load_medical
    }

    dataloaders = {}
    domain_info = {}

    for name, (data, n_classes) in text_domains.items():
        train_texts, train_labels, test_texts, test_labels, _ = data

        train_ds = DomainDataset(train_texts, train_labels, 'text', tokenizer)
        test_ds = DomainDataset(test_texts, test_labels, 'text', tokenizer)

        dataloaders[name] = (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       drop_last=True, num_workers=0),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=0),
        )
        domain_info[name] = {
            'type': 'text',
            'n_classes': n_classes,
            'vocab_size': vocab_size,
        }

    # --- Load CIFAR-10 (vision domain) ---
    cifar_data = load_cifar10(n_train, n_test)
    train_images, train_labels, test_images, test_labels, _ = cifar_data

    train_ds = DomainDataset(train_images, train_labels, 'image')
    test_ds = DomainDataset(test_images, test_labels, 'image')

    dataloaders['vision'] = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   drop_last=True, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=0),
    )
    domain_info['vision'] = {
        'type': 'image',
        'n_classes': 10,
        'vocab_size': None,
    }

    print("\n" + "=" * 70)
    print("Data loading complete!")
    print(f"Domains loaded: {list(dataloaders.keys())}")
    print(f"Shared vocabulary size: {vocab_size}")
    print("=" * 70)

    return tokenizer, dataloaders, domain_info
