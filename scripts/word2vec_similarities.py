import random
from gensim.models import Word2Vec
import pandas as pd

random.seed(42)

true_dataset = pd.read_csv("../dataset/preprocessed_true.csv")
true_dataset.dropna(axis=0, how='any', inplace=True)

fake_dataset = pd.read_csv("../dataset/preprocessed_fake.csv")
fake_dataset.dropna(axis=0, how='any', inplace=True)


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

true_words = [_[0] for _ in true_vocab_count]
fake_words = [_[0] for _ in fake_vocab_count]

word_true_not_in_fake = list(set(true_words).difference(set(fake_words)))
word_fake_not_in_true = list(set(fake_words).difference(set(true_words)))

for k in top_k:
 	model = Word2Vec.load('../model_word2vec/true_fake_word2vec_top_{}.bin'.format(k))
 	vocab_words = list(set(model.wv.vocab))
 	print(k)
 	words_true = [word for word in word_true_not_in_fake if word in vocab_words]
 	words_fake = [word for word in word_fake_not_in_true if word in vocab_words]
 	chosed_words = words_true[:3] + words_fake[:3]
 	for word in chosed_words:
 		print("Word: {}".format(word))
 		result = model.most_similar(positive=[word], topn=3)
 		print(result)
 		print("="*20)