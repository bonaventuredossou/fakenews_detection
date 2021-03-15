from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd


true_dataset = pd.read_csv("../dataset/preprocessed_true.csv")
true_dataset.dropna(axis=0, how='any', inplace=True)

fake_dataset = pd.read_csv("../dataset/preprocessed_fake.csv")
fake_dataset.dropna(axis=0, how='any', inplace=True)

true_corpus = list(true_dataset["text"])
fake_corpus = list(fake_dataset["text"])

sentences_true = [_.split() for _ in true_corpus]
sentences_fake = [_.split() for _ in fake_corpus]

sentences_true_fake = sentences_true + sentences_fake

true_data = [TaggedDocument(d, [i]) for i, d in enumerate(sentences_true)]
fake_data = [TaggedDocument(d, [i]) for i, d in enumerate(sentences_fake)]
true_fake_data = [TaggedDocument(d, [i]) for i, d in enumerate(sentences_true_fake)]

# print(true_data)

## Train doc2vec model
'''
vector_size = Dimensionality of the feature vectors.
window = The maximum distance between the current and predicted word within a sentence.
min_count = Ignores all words with total frequency lower than this.
alpha = The initial learning rate.
'''
model_true_data = Doc2Vec(true_data, vector_size = 20, window = 2, min_count = 1, epochs = 200)
model_fake_data = Doc2Vec(fake_data, vector_size = 20, window = 2, min_count = 1, epochs = 200)
model_true_fake_data = Doc2Vec(true_fake_data, vector_size = 20, window = 2, min_count = 1, epochs = 200)

# save model
model_true_data.save('../model_doc2vec/true_doc2vec.bin')
model_fake_data.save('../model_doc2vec/fake_doc2vec.bin')
model_true_fake_data.save('../model_doc2vec/true_fake_doc2vec.bin')

# # using the model
# test_doc = word_tokenize(sentence.lower())
# test_doc_vector = model.infer_vector(test_doc)
# model.docvecs.most_similar(positive = [test_doc_vector])