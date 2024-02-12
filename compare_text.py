import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from cosine_sklearn import cosine_similarity as similarity_sklearn
from cosine_numpy import cosine_similarity as similarity_numpy
from cosine_torch import cosine_similarity as similarity_torch


Document0 = """laws of aviation
wings are too small
bees don't care"""

Document1 = """According to all known laws of aviation, 
there is no way a bee should be able to fly. 
Its wings are too small to get its fat little body off the ground. 
The bee, of course, flies anyway because 
bees don't care what humans think is impossible."""

Document2 = """Bees are flying insects closely related to wasps and ants, 
known for their role in pollination and, in the case of the best-known 
bee species, the western honey bee, for producing honey. 
Bees are a monophyletic lineage within the superfamily 
Apoidea. They are presently considered a clade, called 
Anthophila. There are over 16,000 known species of bees in seven 
recognized biological families. Some species – 
including honey bees, bumblebees, and stingless bees – 
live socially in colonies while some species – including mason bees, 
carpenter bees, leafcutter bees, and sweat bees – are solitary."""

Document3 = """Data science is an interdisciplinary field that uses 
scientific methods, processes, algorithms and systems to extract 
knowledge and insights from noisy, structured and unstructured data,
and apply knowledge and actionable insights from data across a broad 
range of application domains. Data science is related to data mining, 
machine learning and big data."""

corpus = [Document0, Document1, Document2, Document3]

count_vect = CountVectorizer()
X_train = count_vect.fit_transform(corpus)
count_df = pd.DataFrame(X_train.toarray(),
                        columns=count_vect.get_feature_names_out(),
                        index=['Document 0', 'Document 1', 'Document 2', 'Document 3'])
print(count_df)
# print(similarity_sklearn(count_df.to_numpy()))
# print(similarity_numpy(count_df.to_numpy()))
# print(similarity_torch(torch.Tensor(count_df.to_numpy())))

tfidf_vect = TfidfVectorizer()
X_train = tfidf_vect.fit_transform(corpus)
tfidf_df = pd.DataFrame(X_train.toarray(),
                        columns=tfidf_vect.get_feature_names_out(),
                        index=['Document 0', 'Document 1', 'Document 2', 'Document 3'])
print(tfidf_df)
print(similarity_sklearn(tfidf_df.to_numpy()))
print(similarity_numpy(tfidf_df.to_numpy()))
print(similarity_torch(torch.Tensor(tfidf_df.to_numpy())))
