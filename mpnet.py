"""
https://huggingface.co/sentence-transformers/all-mpnet-base-v2

sentence-transformers/all-mpnet-base-v2 generates 768 dimensions
"""
from transformers import AutoTokenizer, AutoModel
from typing import List
from scipy.signal import argrelextrema
import math
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "Model",
]


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean Pooling - Take attention mask into account for correct averaging
    """
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# def sigmoid(x: float) -> float:
#     return 1.0 / (1.0 + math.exp(0.5 * x))


def cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    """
    https://medium.com/@dhruvbird/all-pairs-cosine-similarity-in-pytorch-867e722c8572
    """

    x_row, x_col = x[None, :, :], x[:, None, :]
    x_row, x_col = x_row.expand(x.shape[0], x.shape[0], 2), x_col.expand(x.shape[0], x.shape[0], 2)
    return F.cosine_similarity(x_row, x_col, dim=-1)


class Model:
    MODEL_CHECKPOINT = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_CHECKPOINT)
        self.model = AutoModel.from_pretrained(self.MODEL_CHECKPOINT)

    def encode(self, sentences: List[str], normalize: bool = True) -> []:
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        if normalize:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()

    def similarities(self, sentences: List[str], normalize: bool = True):
        embeddings = self.encode(sentences, normalize)
        similarities = cosine_similarity(embeddings)
        activated_similarities = activate_similarities(similarities, p_size=5)
        # order parameter controls how frequent should be splits. I would not reccomend changing this parameter.
        minmimas = argrelextrema(activated_similarities, np.less, order=2)


def main():
    model = Model()
    # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted']
    # Output
    print(model.encode(sentences))


if __name__ == "__main__":
    main()
