# Autonomous Audio Explorer: Cross-Species Vocalization Classification

A deep learning system for **cross-domain acoustic classification** capable of recognizing **30 vocalization classes** spanning animal sounds, human healthcare audio, industrial machinery, and environmental events.

The project combines a **Whisper-Tiny encoder**, **Parameter-Efficient Fine-Tuning (LoRA)**, **Domain Adapters**, and a **three-stage training pipeline** to learn robust acoustic representations under severe data scarcity.

---

# Project Overview

Unlike traditional environmental sound classification, this project addresses the challenge of recognizing acoustically diverse sounds across multiple domains using a single model.

The dataset contains **1,092 audio samples** distributed across **30 classes**, averaging only **36 samples per class**, making it a challenging low-resource learning problem.

## Performance Summary

| Metric | Value |
|---------|------:|
| Classes | 30 |
| Domains | 4 |
| Dataset Size | 1,092 samples |
| Validation Accuracy | ~55% |
| Random Baseline | 3.33% |
| Improvement Over Baseline | 16.5× |

The model significantly outperforms random guessing despite extreme class imbalance and limited training data.

---

# Why is this Problem Difficult?

## Cross-Domain Acoustic Variability

The model must distinguish sounds originating from completely different acoustic environments.

Examples include:

- Animal vocalizations
- Human healthcare sounds
- Industrial machinery
- Environmental sounds

Each domain has unique background noise, recording conditions, and spectral characteristics, forcing the model to learn domain-independent acoustic representations instead of memorizing background environments.

---

## Spectral Similarity Between Classes

Many unrelated sounds share similar frequency content.

Examples include:

- Dog whine ↔ Infant cry
- Cat hiss ↔ Dry cough
- Whale song ↔ Siren
- Hydraulic leak ↔ Thunder

Accurate classification therefore requires learning temporal structure and harmonic relationships rather than relying only on spectral appearance.

---

## Low-Resource Learning

With approximately **36 samples per class**, the project operates in a severe low-data regime where conventional deep learning models easily overfit.

To improve generalization, the architecture intentionally limits trainable parameters using:

- Parameter-Efficient Fine-Tuning (LoRA)
- Lightweight domain adapters
- Bottleneck classifier
- Aggressive regularization

---

# Model Architecture

The system uses a pretrained Whisper-Tiny encoder as the feature extractor.

```
Raw Audio
     │
Log-Mel Spectrogram
     │
Whisper-Tiny Encoder
(Frozen + LoRA)
     │
Mean Pooling
     │
Layer Normalization
     │
Domain Adapter
     │
Classifier
384 → 64 → 32 → 30
     │
Predicted Class
```

## Key Components

### Whisper-Tiny Encoder

A pretrained Whisper-Tiny encoder provides robust acoustic representations learned from approximately **680,000 hours** of multilingual speech.

Instead of training the encoder from scratch, the project adapts it to the new task using Parameter-Efficient Fine-Tuning.

---

### LoRA (Parameter-Efficient Fine-Tuning)

Rather than updating every encoder parameter, LoRA inserts low-rank adapters into the attention layers.

Benefits include:

- Reduced trainable parameters
- Lower memory requirements
- Better generalization
- Preservation of pretrained knowledge

LoRA adapters are applied to the attention **query** and **value** projection matrices.

---

### Domain Adapters

Lightweight domain-specific adapters refine pooled Whisper features before classification.

Separate adapters are learned for:

- Animal
- Human Healthcare
- Machinery
- Environmental

This allows the model to capture domain-specific characteristics while preserving a shared acoustic representation.

---

### Bottleneck Classifier

Instead of using a large classifier, the project deliberately compresses the representation:

```
384 → 64 → 32 → 30
```

The bottleneck forces the network to learn compact and generalizable acoustic features instead of memorizing individual samples.

---

# Three-Stage Training Pipeline

Training is divided into three sequential phases.

## Phase 1 – Linear Probing

- Frozen Whisper encoder
- Train only domain adapters and classifier
- Learning rate: 1e-3

Purpose:

Initialize decision boundaries while preserving pretrained representations.

---

## Phase 2 – LoRA Fine-Tuning

- Enable LoRA adapters
- Fine-tune attention layers
- Learning rate: 2e-4

During this phase the model learns task-specific acoustic representations while retaining the knowledge contained in the pretrained Whisper encoder.

A Gradient Reversal Layer (GRL) and domain classifier encourage the encoder to learn domain-invariant features.

---

## Phase 3 – Domain-Specific Fine-Tuning

- Fine-tune using domain-specific subsets
- Learning rate: 5e-5

This final stage specializes each domain adapter while maintaining the shared global representation learned during previous stages.

---

# Repository Structure

```
cross-species-vocalization/

├── models/
├── training/
├── inference/
├── evaluation/
├── data/
├── notebooks/
├── checkpoints/
├── logs/
├── tests/
├── run_training.py
├── requirements.txt
└── README.md
```

## Main Components

| Directory | Description |
|------------|-------------|
| models | Neural network architecture and configuration |
| training | Dataset loading, augmentations, Lightning modules and training pipeline |
| inference | Offline and real-time audio classification |
| evaluation | Metrics, benchmarking and confusion matrix generation |
| data | Dataset downloads and processed splits |
| checkpoints | Saved model weights |
| notebooks | Data exploration and error analysis |

---

# Getting Started

## Clone the Repository

```bash
git clone https://github.com/Mayenmein/cross-species-vocalization.git

cd cross-species-vocalization
```

## Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

Linux/macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

The training pipeline automatically downloads data from:

- ESC-50
- Freesound

After preprocessing, the dataset is split into:

- Training
- Validation
- Testing

---

# Training

Run the complete three-stage training pipeline:

```bash
python run_training.py
```

The pipeline performs:

1. Linear probing
2. Global LoRA adaptation
3. Domain-specific fine-tuning

Checkpoints and logs are saved automatically.

---

# Inference

Single-file inference and real-time microphone classification are supported through the `inference` module.

The inference pipeline performs:

- Audio loading
- Feature extraction
- Whisper encoding
- Domain adaptation
- Class prediction

---

# Results

The final model demonstrates that pretrained speech representations can successfully transfer to cross-domain acoustic classification, even under severe data scarcity.

Key achievements include:

- 30-class classification
- Four independent acoustic domains
- Approximately 55% validation accuracy
- 16.5× improvement over random baseline
- Parameter-efficient adaptation using LoRA
- Multi-stage training with domain-adversarial learning

---

# Future Improvements

Potential future work includes:

- Larger curated datasets
- Stronger Whisper backbones
- Hierarchical classification
- Additional acoustic domains
- Improved real-time deployment
- Expanded benchmarking against public datasets

---

# Technologies

- Python
- PyTorch
- PyTorch Lightning
- Hugging Face Transformers
- PEFT (LoRA)
- OpenAI Whisper
- Weights & Biases