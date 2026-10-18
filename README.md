# SentimentScope: Transformer-Based Sentiment Analysis

**CineScope Machine Learning Engineering Project**  
*AWS AI & ML Scholarship / Udacity Nanodegree Program*

---

## 🌟 Overview

CineScope aims to enhance its movie and show recommendation system by extracting audience sentiment from user reviews. **SentimentScope** implements and trains a customized transformer neural network architecture (`DemoGPT`) from scratch to perform binary sentiment classification (Positive vs. Negative) on the benchmark IMDB dataset.

### Key Highlights
- **Architecture**: 6-layer Transformer Decoder with 8 multi-head self-attention heads (`d_embed=256`, `head_size=32`, context size 128).
- **Sequence Classification Adaptation**: Mean pooling over token representations connected to a 2-class linear classification head.
- **Subword Tokenization**: Pre-trained `bert-base-uncased` subword tokenizer from Hugging Face.
- **Performance**: Achieved **76.93%** test accuracy on the held-out IMDB test set (exceeding the >75% rubric benchmark).
- **Production-Ready Inference**: Standalone inference module (`inference.py`) and class interface for instant batch evaluation.

---

## 📁 Repository Structure

```
aws_scholer_project/
├── SentimentScope.ipynb          # Fully executed Jupyter notebook with all code, tests, and outputs
├── SentimentScope_starter.ipynb  # Starter notebook format with all cells completed
├── best_model.pt                 # Saved model checkpoint (achieves >76.9% test accuracy)
├── inference.py                  # Standalone inference script & SentimentPredictor class
├── aclImdb_v1.tar.gz             # IMDB raw dataset archive
├── README.md                     # Project documentation & summary
└── submission.zip                # Submission package containing all deliverables
```

---

## 🛠️ Model Architecture & Technical Details

```
Input Text Review
       │
       ▼
AutoTokenizer ('bert-base-uncased')  --> max_length = 128
       │
       ▼
Token Embeddings (vocab_size=30,522, d_embed=256) + Learned Positional Embeddings (128, 256)
       │
       ▼
6x Transformer Blocks:
   ├── LayerNorm -> MultiHeadAttention (8 heads, head_size=32) + Residual
   └── LayerNorm -> FeedForward (256 -> 1024 -> 256, GELU, Dropout 0.2) + Residual
       │
       ▼
Layer Normalization (256)
       │
       ▼
Mean Pooling across Sequence Dimension: (Batch, 128, 256) -> (Batch, 256)
       │
       ▼
Linear Classification Head: (256 -> 2 logits)
       │
       ▼
Softmax -> [P(Negative), P(Positive)]
```

### Hyperparameters Config
| Parameter | Value | Description |
|---|---|---|
| `vocabulary_size` | 30,522 | Hugging Face `bert-base-uncased` vocabulary |
| `num_classes` | 2 | Binary sentiment (0: Negative, 1: Positive) |
| `d_embed` | 256 | Model embedding and hidden dimension |
| `context_size` | 128 | Maximum sequence token length |
| `layers_num` | 6 | Number of transformer decoder blocks |
| `heads_num` | 8 | Multi-head self-attention heads |
| `head_size` | 32 | Dimension per attention head (8 * 32 = 256) |
| `dropout_rate` | 0.2 | Regularization dropout rate |
| `learning_rate` | 3e-4 | AdamW optimizer learning rate |
| `batch_size` | 32 | Mini-batch size |
| `epochs` | 8 | Full training epochs |

---

## 📊 Training & Evaluation Results

| Epoch | Training Loss | Validation Accuracy | Notes |
|---|---|---|---|
| Baseline | N/A | 49.24% | Initial random weights |
| Epoch 1 | 0.6970 | 56.40% | Model starts capturing polarity |
| Epoch 2 | 0.5898 | 65.12% | Attention patterns stabilizing |
| Epoch 4 | 0.4321 | 72.80% | Consistent convergence |
| Epoch 6 | 0.3150 | 77.20% | Surpassed project benchmark |
| Epoch 8 | 0.2014 | 80.08% | Peak validation accuracy |
| **Test Set** | **N/A** | **76.93%** | **Passed (>75% Rubric Requirement)** |

---

## 🚀 How to Run Inference

### 1. Using Python
Run the standalone inference script:
```bash
python inference.py
```

### 2. Using the `SentimentPredictor` in Code
```python
from inference import SentimentPredictor

predictor = SentimentPredictor("best_model.pt")

reviews = [
    "This movie was an absolute masterpiece! Brilliant acting and direction.",
    "Terrible film. The plot was completely boring and a waste of time."
]

predictions = predictor.predict(reviews)
for p in predictions:
    print(f"Text: {p['text']}")
    print(f"Sentiment: {p['label']} (Confidence: {p['confidence']*100:.2f}%)\n")
```

---

## 📋 Rubric Compliance Checklist

- [x] **Data Loading Helper Function**: `load_dataset()` loads text files and builds DataFrames.
- [x] **Dataframe Assertions**: Validated 25,000 rows and 2 columns for both train and test splits.
- [x] **Descriptive Statistics & Visualizations**: Included summary statistics, label distribution bar plot, and review length histogram.
- [x] **Custom PyTorch Dataset (`IMDBDataset`)**: Implemented `__init__`, `__len__`, and `__getitem__` with tokenization, padding, and truncation.
- [x] **Dataset Assertions**: Confirmed train (22,500), val (2,500), test (25,000) splits and tensor formats.
- [x] **Custom Transformer Architecture (`DemoGPT`)**: Implemented mean pooling and linear classification head.
- [x] **Model Output Assertion**: Output shape verified as `(32, 2)`.
- [x] **Accuracy Calculation**: `calculate_accuracy()` implemented using evaluation mode and `torch.no_grad()`.
- [x] **Training Loop**: Completed using cross-entropy loss, AdamW optimizer, and validation tracking.
- [x] **Test Set Accuracy**: Achieved **76.93%**, exceeding the 75% requirement.
- [x] **Model Checkpoint**: Saved to `best_model.pt`.
- [x] **Standout Feature**: Provided `SentimentPredictor` class and `inference.py` for batch inference.
