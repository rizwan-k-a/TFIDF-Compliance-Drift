import numpy as np
from sklearn.metrics.pairwise import linear_kernel


def cosine_similarity_matrix(X):
    # X is a sparse matrix or dense 2D array
    return linear_kernel(X, X)


def pairwise_similarity(doc_vecs, ref_vecs):
    # returns matrix (n_docs, n_refs)
    return linear_kernel(doc_vecs, ref_vecs)
