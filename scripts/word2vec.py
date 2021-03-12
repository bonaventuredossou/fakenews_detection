import random
from gensim.models import Word2Vec
from matplotlib import pyplot
from sklearn.decomposition import PCA
import pandas as pd

random.seed(42)

true_dataset = pd.read_csv("../dataset/preprocessed_true.csv")
true_dataset.dropna(axis=0, how='any', inplace=True)

fake_dataset = pd.read_csv("../dataset/preprocessed_fake.csv")
fake_dataset.dropna(axis=0, how='any', inplace=True)

true_corpus = list(true_dataset["text"])
fake_corpus = list(fake_dataset["text"])

sentences_true = [_.split() for _ in true_corpus]
sentences_fake = [_.split() for _ in fake_corpus]

# train model
model_true = Word2Vec(sentences_true, min_count=1, alpha=0.025)
model_fake = Word2Vec(sentences_fake, min_count=1, alpha=0.025)

# save model
model_true.save('../model_word2vec/true_word2vec.bin')
model_fake.save('../model_word2vec/fake_word2vec.bin')

# load model
new_model_true = Word2Vec.load('../model_word2vec/true_word2vec.bin')
new_model_fake = Word2Vec.load('../model_word2vec/fake_word2vec.bin')

X = new_model_true[new_model_true.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(set(new_model_true.wv.vocab))
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.title("Visualization of Real Article Word Embedding - Word Vectors Using PCA")
pyplot.savefig("../plots/word2vec_true.png")
# pyplot.show()



Y = new_model_fake[new_model_fake.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(Y)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(set(new_model_fake.wv.vocab))
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.title("Visualization of Fake Article Word Embedding - Word Vectors Using PCA")
pyplot.savefig("../plots/word2vec_fake.png")
# pyplot.show()
