"""
SentimentScope: Transformer Sentiment Analysis Inference Module
CineScope Recommendation System

This module provides an easy-to-use interface to load the trained DemoGPT
sentiment analysis model and classify movie reviews as Positive or Negative.
"""

import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer


class AttentionHead(nn.Module):
    """Single causal self-attention head."""
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.K_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.V_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.dropout = nn.Dropout(config["dropout_rate"])
        casual_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer('casual_attention_mask', casual_attention_mask)

    def forward(self, input):
        batch_size, tokens_num, d_embed = input.shape
        Q = self.Q_weights(input)
        K = self.K_weights(input)
        V = self.V_weights(input)

        attention_scores = Q @ K.transpose(1, 2)
        attention_scores = attention_scores.masked_fill(
            self.casual_attention_mask[:tokens_num, :tokens_num] == 0,
            float('-inf')
        )
        attention_scores = attention_scores / math.sqrt(K.shape[-1])
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        return attention_scores @ V


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention composed of multiple AttentionHead modules."""
    def __init__(self, config):
        super().__init__()
        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
        self.heads = nn.ModuleList(heads_list)
        self.linear = nn.Linear(config["heads_num"] * config["head_size"], config["d_embed"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, input):
        heads_outputs = [head(input) for head in self.heads]
        x = torch.cat(heads_outputs, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""
    def __init__(self, config):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(config["d_embed"], 4 * config["d_embed"]),
            nn.GELU(),
            nn.Linear(4 * config["d_embed"], config["d_embed"]),
            nn.Dropout(config["dropout_rate"])
        )

    def forward(self, input):
        return self.linear_layers(input)


class Block(nn.Module):
    """Transformer decoder block with residual connections and pre-layer normalization."""
    def __init__(self, config):
        super().__init__()
        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["d_embed"])
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["d_embed"])

    def forward(self, input):
        x = input
        x = x + self.multi_head(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class DemoGPT(nn.Module):
    """
    DemoGPT Transformer customized for binary sentiment classification.
    Uses mean pooling over token embeddings followed by a linear classification head.
    """
    def __init__(self, config):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["d_embed"])
        self.positional_embedding_layer = nn.Embedding(config["context_size"], config["d_embed"])
        blocks = [Block(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)
        self.layer_norm = nn.LayerNorm(config["d_embed"])
        self.classifier = nn.Linear(config["d_embed"], config["num_classes"], bias=False)

    def forward(self, token_ids):
        batch_size, tokens_num = token_ids.shape
        x = self.token_embedding_layer(token_ids)
        positions = torch.arange(tokens_num, device=token_ids.device)
        pos_embed = self.positional_embedding_layer(positions)
        x = x + pos_embed.unsqueeze(0)
        x = self.layers(x)
        x = self.layer_norm(x)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


class SentimentPredictor:
    """
    High-level inference wrapper for sentiment analysis.
    Supports single-string predictions and batch inference.
    """
    def __init__(self, checkpoint_path="best_model.pt", tokenizer_name="bert-base-uncased", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        self.test_accuracy = checkpoint.get('test_accuracy', None)

        self.model = DemoGPT(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = self.config.get("context_size", 128)

    def predict(self, texts):
        """
        Classifies sentiment for a text string or list of text strings.

        Args:
            texts (str or list of str): Input review text(s).

        Returns:
            list of dict: Prediction results with labels, confidence, and class probabilities.
        """
        if isinstance(texts, str):
            texts = [texts]

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encodings['input_ids'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)

        results = []
        for i, text in enumerate(texts):
            pred_class = predicted_classes[i].item()
            prob = probabilities[i][pred_class].item()
            label = "Positive" if pred_class == 1 else "Negative"
            results.append({
                "text": text,
                "label": label,
                "confidence": prob,
                "prob_positive": probabilities[i][1].item(),
                "prob_negative": probabilities[i][0].item()
            })
        return results


if __name__ == '__main__':
    print("Initializing SentimentPredictor from best_model.pt...")
    predictor = SentimentPredictor("best_model.pt")
    if predictor.test_accuracy:
        print(f"Loaded checkpoint with verified test accuracy: {predictor.test_accuracy:.2f}%")

    test_reviews = [
        "This movie was an absolute masterpiece! Brilliant acting and direction.",
        "Terrible film. The plot was completely boring and a waste of time.",
        "A visually stunning and deeply moving cinematic experience.",
        "Poor screenplay, dull characters, and unconvincing performances."
    ]

    print("\n--- Running Inference on Sample Reviews ---")
    predictions = predictor.predict(test_reviews)
    for res in predictions:
        print(f"Review: \"{res['text']}\"")
        print(f"Prediction: {res['label']} ({res['confidence'] * 100:.2f}% confidence)")
        print(f"Details: P(Pos)={res['prob_positive']:.4f}, P(Neg)={res['prob_negative']:.4f}\n")
