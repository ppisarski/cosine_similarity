"""
https://medium.com/@npolovinkin/how-to-chunk-text-into-paragraphs-using-python-8ae66be38ea6

https://medium.com/@nils_reimers/openai-gpt-3-text-embeddings-really-a-new-state-of-the-art-in-dense-text-embeddings-6571fe3ec9d9

"""

import numpy as np
import json
import pysbd
import re
from sentence_transformers import SentenceTransformer
# from cosine_torch import cosine_similarity, cosine_similarity_numpy
from cosine_numpy import cosine_similarity
from scipy.signal import argrelextrema
import seaborn as sns
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(0.5 * x))


def activate_similarities(similarities: np.array, p_size: int = 10) -> np.array:
    """ Function returns list of weighted sums of activated sentence similarities
    Args:
        similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
        p_size (int): number of sentences are used to calculate weighted sum
    Returns:
        list: list of weighted sums
    """
    # To create weights for sigmoid function we first have to create space.
    # P_size will determine number of sentences used and the size of weights vector.
    x = np.linspace(-10, 10, p_size)
    # Then we need to apply activation function to the created space.
    y = np.vectorize(sigmoid)
    print(list(y(x)))
    # Because we only apply activation to p_size number of sentences we have to add zeros
    # to neglect the effect of every additional sentence and to match the length of vector we will multiply.
    activation_weights = np.pad(y(x), (0, similarities.shape[0] - p_size))
    # 1. Take each diagonal to the right of the main diagonal.
    diagonals = [similarities.diagonal(each) for each in range(0, similarities.shape[0])]
    # 2. Pad each diagonal by zeros at the end.
    # Because each diagonal is different length we should pad it with zeros at the end.
    diagonals = [np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals]
    # 3. Stack those diagonals into new matrix.
    diagonals = np.stack(diagonals)
    # 4. Apply activation weights to each row. Multiply similarities with our activation.
    diagonals = diagonals * activation_weights.reshape(-1, 1)
    # 5. Calculate the weighted sum of activated similarities.
    activated_similarities = np.sum(diagonals, axis=0)
    return activated_similarities


def split_similarities(similarities: np.array, p_size: int = 10) -> np.array:
    # Lets apply our function. For long sentences i recommend to use 10 or more sentences
    activated_similarities = activate_similarities(similarities, p_size=p_size)
    # 6. Find relative minima of our vector.
    # order parameter controls how frequent should be splits.
    minimas = argrelextrema(activated_similarities, np.less, order=1)
    split_points = minimas[0]
    return split_points


def cleanup_text(text: str) -> str:
    """
    Text cleanup.
    :param text:
    :return: text
    """
    # identify paragraphs
    paragraphs = text.split("\n\n")
    # remove double spaces, keep paragraphs with numbers only
    paragraphs = [" ".join(paragraph.split()) for paragraph in paragraphs
                  if re.search('[a-zA-Z]', paragraph)]
    return "\n".join(paragraphs)


def split_paragraphs(text: str) -> list:
    # identify paragraphs
    paragraphs = text.split("\n")
    # remove double spaces, etc.
    paragraphs = [" ".join(paragraph.split()) for paragraph in paragraphs]
    return paragraphs


def main():
    with open("data/pg2701.txt", "r") as f:
        text = f.read()
        text = cleanup_text(text)

    paragraphs = split_paragraphs(text)
    print("paragraphs: ", len(paragraphs))

    sentences = []
    seg = pysbd.Segmenter(language="en", clean=False)
    for paragraph in paragraphs:
        sentences += seg.segment(paragraph)
    sentences = [sentence.strip() for sentence in sentences]
    print("sentences: ", len(sentences))

    # with open("data/sentences.json", "w") as f:
    #     json.dump(sentences, f)

    sentences = sentences[:1000]

    # Get the length of each sentence
    sentence_token_count = np.array([len(sentence.split()) for sentence in sentences])
    print("min", np.min(sentence_token_count),
          len([c for c in sentence_token_count if c <= np.min(sentence_token_count)]))
    print("P1", np.percentile(sentence_token_count, 1),
          len([c for c in sentence_token_count if c <= np.percentile(sentence_token_count, 1)]))
    print("P10", np.percentile(sentence_token_count, 10),
          len([c for c in sentence_token_count if c <= np.percentile(sentence_token_count, 10)]))
    print("P50", np.percentile(sentence_token_count, 50),
          len([c for c in sentence_token_count if c <= np.percentile(sentence_token_count, 50)]))
    print("P90", np.percentile(sentence_token_count, 90),
          len([c for c in sentence_token_count if c <= np.percentile(sentence_token_count, 90)]))
    print("P99", np.percentile(sentence_token_count, 99),
          len([c for c in sentence_token_count if c <= np.percentile(sentence_token_count, 99)]))
    print("max", np.max(sentence_token_count),
          len([c for c in sentence_token_count if c <= np.max(sentence_token_count)]))

    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(sentences)
    # print(embeddings.shape)
    # Normalize the embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    similarities = cosine_similarity(embeddings)
    # print(similarities.shape)
    # sns.heatmap(similarities).set_title('Cosine similarities matrix')
    # plt.show()
    split_points = split_similarities(similarities, 5)

    paragraphs = []
    for start, end in np.dstack((np.insert(split_points[:-1], 0, 0), split_points))[0]:
        paragraphs += " ".join(sentences[start:end]),

    # for idx, sentence in enumerate(sentences):
    #     if idx in split_points:
    #         paragraphs += f'\n\n{sentence}'
    #     else:
    #         text += f'{sentence} '

    with open("data/out.json", "w") as f:
        json.dump(paragraphs, f)

    # Get the length of each paragraph
    paragraph_token_count = np.array([len(paragraph.split()) for paragraph in paragraphs])
    print("min", np.min(paragraph_token_count),
          len([c for c in paragraph_token_count if c <= np.min(paragraph_token_count)]))
    print("P1", np.percentile(paragraph_token_count, 1),
          len([c for c in paragraph_token_count if c <= np.percentile(paragraph_token_count, 1)]))
    print("P10", np.percentile(paragraph_token_count, 10),
          len([c for c in paragraph_token_count if c <= np.percentile(paragraph_token_count, 10)]))
    print("P50", np.percentile(paragraph_token_count, 50),
          len([c for c in paragraph_token_count if c <= np.percentile(paragraph_token_count, 50)]))
    print("P90", np.percentile(paragraph_token_count, 90),
          len([c for c in paragraph_token_count if c <= np.percentile(paragraph_token_count, 90)]))
    print("P99", np.percentile(paragraph_token_count, 99),
          len([c for c in paragraph_token_count if c <= np.percentile(paragraph_token_count, 99)]))
    print("max", np.max(paragraph_token_count),
          len([c for c in paragraph_token_count if c <= np.max(paragraph_token_count)]))


if __name__ == "__main__":
    main()
