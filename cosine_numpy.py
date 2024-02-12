"""

"""
import numpy as np
import warnings

# def cosine_similarity(x: np.array, y: np.array) -> np.array:
#     return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def cosine_similarity(x: np.array) -> np.array:
    """
    https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # base similarity matrix (all dot products)
        # replace this with A.dot(A.T).toarray() for sparse representation
        similarity = np.dot(x, x.T)
        # squared magnitude of preference vectors (number of occurrences)
        square_mag = np.diag(similarity)
        # inverse squared magnitude
        inv_square_mag = 1 / square_mag
        # if it doesn't occur, set its inverse magnitude to zero (instead of inf)
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        # inverse of the magnitude
        inv_mag = np.sqrt(inv_square_mag)
        # cosine similarity (elementwise multiply by inverse magnitudes)
        similarity = similarity * inv_mag
        similarity = similarity.T * inv_mag
    return similarity


def main():
    x = np.array([[0, 0.1, 0.2], [1.1, 1.2, 1]])
    similarity = cosine_similarity(x)
    print(similarity)


if __name__ == "__main__":
    main()
