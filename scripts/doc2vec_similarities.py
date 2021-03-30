from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
from nltk.tokenize import word_tokenize
import random

random.seed(42)

true_dataset = pd.read_csv("../dataset/preprocessed_true.csv")
true_dataset.dropna(axis=0, how='any', inplace=True)

fake_dataset = pd.read_csv("../dataset/preprocessed_fake.csv")
fake_dataset.dropna(axis=0, how='any', inplace=True)

true_corpus = list(true_dataset["text"])
fake_corpus = list(fake_dataset["text"])

sentences_true = [_.split() for _ in true_corpus]
sentences_fake = [_.split() for _ in fake_corpus]

sentences_true_fake = sentences_true + sentences_fake

true_uniq = list(set(true_corpus).difference(set(fake_corpus)))
fake_uniq = list(set(fake_corpus).difference(set(true_corpus)))

random.shuffle(true_uniq)
random.shuffle(fake_uniq)

model = Doc2Vec.load('../model_doc2vec/true_fake_doc2vec.bin')

true_sentence = true_uniq[0]
fake_sentence = fake_uniq[0]

print(true_sentence)
print(fake_sentence)

true_test_doc = word_tokenize(true_sentence.lower())
fake_test_doc = word_tokenize(fake_sentence.lower())

true_test_doc_vector = model.infer_vector(true_test_doc)
fake_test_doc_vector = model.infer_vector(fake_test_doc)

true_result = model.docvecs.most_similar(positive = [true_test_doc_vector], topn=3)
fake_result = model.docvecs.most_similar(positive = [fake_test_doc_vector], topn=3)


print("="*20, "Real", "="*20)
for i,j in true_result:
    print("Sentence: {} - Similarity score: {}".format(' '.join(i.strip() for i in sentences_true_fake[i]),j))
print("="*20, "Fake", "="*20)
for i,j in fake_result:
    print("Sentence: {} - Similarity score: {}".format(' '.join(i.strip() for i in sentences_true_fake[i]),j))
print("="*20, "End", "="*20)
