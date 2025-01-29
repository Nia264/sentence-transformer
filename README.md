# Multitask Learning with Transformer

This repository contains a PyTorch implementation of a multitask learning model using a Transformer architecture. The model is designed to handle two tasks simultaneously: **Sentence Classification** and **Named Entity Recognition (NER)**. The implementation includes the model architecture, training loop, and example usage.

## Overview

The multitask learning model is built on top of a Transformer encoder and includes task-specific heads for sentence classification and NER. The model is trained using a shared encoder for both tasks, allowing it to learn generalizable features that benefit both tasks.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Model Initialization

To initialize the `MultitaskTransformer` model:

```python
from multitask_learning import MultitaskTransformer

model = MultitaskTransformer(
    vocab_size=1000,  # Size of the vocabulary
    num_classes=3,    # Number of classification classes
    num_ner_tags=5,   # Number of NER tags
    d_model=512,      # Embedding dimension
    nhead=8,          # Number of attention heads
    num_encoder_layers=6,  # Number of transformer encoder layers
    dim_feedforward=2048,  # Dimension of the feedforward network
    dropout=0.1,      # Dropout probability
    max_seq_length=512  # Maximum sequence length
)
```

### Training

To train the model, use the `train_model` function from `trainer.py`:

```python
from trainer import train_model, MultitaskDataset

# Example datasets
train_dataset = MultitaskDataset(texts, classification_labels, ner_labels, tokenizer=None)
val_dataset = MultitaskDataset(val_texts, val_classification_labels, val_ner_labels, tokenizer=None)

# Train the model
history = train_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=10,
    learning_rate=2e-5,
    warmup_steps=1000,
    weight_decay=0.01
)
```

### Inference

To perform inference with the trained model:

```python
# For classification
outputs = model(src, task='classification', src_padding_mask=padding_mask)
classification_logits = outputs['logits']

# For NER
outputs = model(src, task='ner', src_padding_mask=padding_mask)
ner_logits = outputs['logits']
```

## Model Architecture

The model consists of the following components:

1. **Transformer Encoder**: Shared across both tasks, it processes the input sequence and generates contextualized embeddings.
2. **Sentence Classification Head**: A task-specific head that takes the pooled output from the encoder and predicts the classification label.
3. **NER Head**: A task-specific head that predicts NER tags for each token in the input sequence.

### Loss Function

The model uses a multitask loss function that combines the losses for both tasks:

- **CrossEntropyLoss** for sentence classification.
- **CrossEntropyLoss** with `ignore_index=-100` for NER (to ignore padding tokens).

## Training

The training loop is implemented in the `MultitaskTrainer` class, which handles:

- Randomly selecting a task for each batch.
- Calculating the loss for the selected task.
- Updating the model parameters using gradient descent.
- Evaluating the model on the validation set.

### Metrics

During training and evaluation, the following metrics are tracked:

- **Training Loss**: Average loss across all training batches.
- **Validation Loss**: Separate losses for classification and NER tasks.
- **Accuracy**: Classification accuracy and NER accuracy (excluding padding tokens).

## Example

An example of how to train and evaluate the model is provided in the `test_training` function in `trainer.py`. This function creates dummy datasets, initializes the model, and trains it for a few epochs.

```python
from trainer import test_training

# Run the example training loop
history = test_training()
```
