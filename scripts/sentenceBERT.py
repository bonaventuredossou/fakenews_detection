import random
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

random.seed(42)

true_dataset = pd.read_csv("../dataset/preprocessed_true.csv")
true_dataset.dropna(axis=0, how='any', inplace=True)

fake_dataset = pd.read_csv("../dataset/preprocessed_fake.csv")
fake_dataset.dropna(axis=0, how='any', inplace=True)

true_corpus = list(true_dataset["text"])
fake_corpus = list(fake_dataset["text"])

true_fake_corpus = true_corpus + fake_corpus
random.shuffle(true_fake_corpus)

sentence_bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

true_sentence_embeddings = sentence_bert_model.encode(true_corpus)
fake_sentence_embeddings = sentence_bert_model.encode(fake_corpus)
true_fake_sentence_embeddings = sentence_bert_model.encode(true_fake_corpus)

np.save("../sentenceBertModels/true_bertEmbedding", true_sentence_embeddings)
np.save("../sentenceBertModels/fake_bertEmbedding", fake_sentence_embeddings)
np.save("../sentenceBertModels/true_fake_bertEmbedding", true_fake_sentence_embeddings)

print('True BERT embedding vector - length', len(true_sentence_embeddings[0]))
print('Fake BERT embedding vector - length', len(fake_sentence_embeddings[0]))
print('True_Fake BERT embedding vector - length', len(true_fake_sentence_embeddings[0]))

# True BERT embedding vector - length 768
# Fake BERT embedding vector - length 768
# True_Fake BERT embedding vector - length 768


# For testing

# query = "Some sentences"
# query_vec = model.encode([query])[0]

# for sent in sentences:
#   sim = cosine(query_vec, model.encode([sent])[0])
#   print("Sentence = ", sent, "; similarity = ", sim)