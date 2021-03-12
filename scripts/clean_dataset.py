# Bonaventure F. P. Dossou - MSc in Data Engineering
# Project: RL Approach in Fake News Detection
# Task at hand in this script: Clean a (.csv) dataframe (dataset)

# imports
import pandas as pd
import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
# from geograpy import places

true_file_name = "../dataset/True.csv"
fake_file_name = "../dataset/Fake.csv"

true_dataset = pd.read_csv(true_file_name)
true_dataset.dropna(axis=0, how='any', inplace=True)

fake_dataset = pd.read_csv(fake_file_name)
fake_dataset.dropna(axis=0, how='any', inplace=True)

punctuations = string.punctuation
punctuations = punctuations.replace("-", "")


def explode_characters(article):
    """
    Task: Explode abbreviations
    Args: article: an article
    returns: the article with the abbreviations expanded in a proper form
    """
    article = re.sub('<u>', '', article)
    article = re.sub('</u>', '', article)
    article = re.sub('\[', '', article)
    article = re.sub('\]', '', article)
    article = re.sub('\^', '', article)
    article = re.sub('\#', '', article)
    article = re.sub('\$', '', article)
    article = re.sub('\*', '', article)
    article = re.sub(" ’ ", "'", article)
    article = re.sub(r"i'm", "i am", article)
    article = re.sub(r"he's", "he is", article)
    article = re.sub(r"she's", "she is", article)
    article = re.sub(r"it's", "it is", article)
    article = re.sub(r"that's", "that is", article)
    article = re.sub(r"there's", "there is", article)
    article = re.sub(r"what's", "what is", article)
    article = re.sub(r"where's", "where is", article)
    article = re.sub(r"how's", "how is", article)
    article = re.sub(r"\'ll", " will", article)
    article = re.sub(r"\'ve", " have", article)
    article = re.sub(r"\'ve", " have", article)
    article = re.sub(r"\'re", " are", article)
    article = re.sub(r"\'d", " would", article)
    article = re.sub(r"\'re", " are", article)
    article = re.sub(r"won't", "will not", article)
    article = re.sub(r"can't", "cannot", article)
    article = re.sub(r"n't", " not", article)
    article = re.sub(r"n'", "ing", article)
    article = re.sub(r"'bout", "about", article)
    article = re.sub(r"'til", "until", article)
    article = re.sub(r"[()\"#/@;:<>{}`+=~|.!?,]", "", article)

    return article


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(article_):
    """
    Task: preprocessed the exploded article by removing punctuations, non-ascii characters etc.
    Args: article exploded from explode_characters()
    return: cleaned and preprocessed article
    """
    article_ = article_.lower().strip()
    article_ = explode_characters(article_)
    
    article_ = re.sub(r"([?.!,¿])", r" \1 ", article_)
    article_ = re.sub(r"[^a-zA-Z.!?]+", r" ", article_)
    article_ = re.sub(r"\s+", r" ", article_).strip()
    article_ = re.sub(r'[" "]+', " ", article_)

    article_ = ' '.join([word for word in article_.split() if word.isalpha()])

    # remove stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(article_)
    article_ = ' '.join(w.strip() for w in word_tokens if w not in stop_words)

    return article_

# Before pre-processing
print("Real articles dataset before cleaning")
print(true_dataset['text'].head(5))
print("\n")
print("Fake articles dataset before cleaning")
print(fake_dataset['text'].head(5))
print("\n")

true_dataset['text'] = true_dataset['text'].apply(preprocess_sentence)
fake_dataset['text'] = fake_dataset['text'].apply(preprocess_sentence)

# After pre-processing
print("Real articles dataset after cleaning")
print(true_dataset['text'].head(5))
print("\n")
print("Fake articles dataset after cleaning")
print(fake_dataset['text'].head(5))
print("\n")
# saving the entire preprocessed dataset
true_dataset.to_csv("../dataset/preprocessed_true.csv", index=False)
fake_dataset.to_csv("../dataset/preprocessed_fake.csv", index=False)