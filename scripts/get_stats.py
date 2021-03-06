import random
from gensim.models import Word2Vec
from matplotlib import pyplot
from sklearn.decomposition import PCA
import pandas as pd
class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


true_dataset = pd.read_csv("../dataset/preprocessed_true.csv")
true_dataset.dropna(axis=0, how='any', inplace=True)

fake_dataset = pd.read_csv("../dataset/preprocessed_fake.csv")
fake_dataset.dropna(axis=0, how='any', inplace=True)

true_corpus = list(true_dataset["text"])
fake_corpus = list(fake_dataset["text"])

true_vocab = Lang()
fake_vocab = Lang()

for i in true_corpus:
    true_vocab.addSentence(i)

for i in fake_corpus:
    fake_vocab.addSentence(i)


top_k = [200, 500, 1000, 1500, 2000]

true_vocab_count = sorted(true_vocab.word2count.items(), key=lambda x: x[1], reverse=True)
fake_vocab_count = sorted(fake_vocab.word2count.items(), key=lambda x: x[1], reverse=True)


for k in top_k:

    n_frequent_true = list(_[0] for _ in true_vocab_count)[:k]
    n_frequent_fake = list(_[0] for _ in fake_vocab_count)[:k]

    words_true = [[_.strip()] for _ in n_frequent_true]
    words_fake = [[_.strip()] for _ in n_frequent_fake]
    words_true_fake = words_true+words_fake
    # train model
    model_true = Word2Vec(words_true, min_count=1, alpha=0.025)
    model_fake = Word2Vec(words_fake, min_count=1, alpha=0.025)
    model_true_fake = Word2Vec(words_true_fake, min_count=1, alpha=0.025)

    # save model
    model_true.save('../model_word2vec/true_word2vec_top_{}.bin'.format(k))
    model_fake.save('../model_word2vec/fake_word2vec_top_{}.bin'.format(k))
    model_true_fake.save('../model_word2vec/true_fake_word2vec_top_{}.bin'.format(k))
    
    # load model
    new_model_true = Word2Vec.load('../model_word2vec/true_word2vec_top_{}.bin'.format(k))
    new_model_fake = Word2Vec.load('../model_word2vec/fake_word2vec_top_{}.bin'.format(k))
    new_model_true_fake = Word2Vec.load('../model_word2vec/true_fake_word2vec_top_{}.bin'.format(k))

    X = new_model_true[new_model_true.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(set(new_model_true.wv.vocab))
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.title("Visualization of Real Article Word Embedding - Word Vectors Using PCA")
    pyplot.savefig("../plots/word2vec_true_top_{}.png".format(k))
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
    pyplot.savefig("../plots/word2vec_fake_top_{}.png".format(k))
    # pyplot.show()
    
    Z = new_model_true_fake[new_model_true_fake.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(Z)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(set(new_model_true_fake.wv.vocab))
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.title("Visualization of Rreal-Fake Article Word Embedding - Word Vectors Using PCA")
    pyplot.savefig("../plots/word2vec_true_fake_top_{}.png".format(k))
    # pyplot.show()