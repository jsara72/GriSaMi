# author: jsara72
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
#from sklearn.model_selection import GridSearchCV

import csv
import pickle
from io import StringIO
import random
import numpy as np
import pandas as pd
import re

# For Python3 uncomment:
#from statistics import mode
# For Python 2:
from scipy.stats import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        # Fix this for Python 3 statistics
        return mode(votes)[0][0]

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        # Fix this for Python 3 statistics
        choice_votes = votes.count(mode(votes)[0][0])
        conf = choice_votes / len(votes)
        return conf

def find_features(document, word_features):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def movie_reviews_documents():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    print("movie_reviews.words(fileid): ", movie_reviews.fileids(category)[:2])
    print("movie_reviews.categories(): ", movie_reviews.categories())
    print("words: ", movie_reviews.words()[0])
    all_words = []
    for w in movie_reviews.words():
            all_words.append(w.lower())

    return documents, all_words


def sentiment_analysis_dataset_documents(reload=True):
    if reload:
        documents = pickle.load(open("documents.pkl","rb"))
        all_words = pickle.load(open("all_words.pkl","rb"))
        return documents, all_words

    print("Reading dataset...")
    dataset = pd.read_csv("Sentiment Analysis Dataset_1000.csv", quotechar='"', quoting=0,  error_bad_lines=False)
    categories = dataset['Sentiment']
    sentences = dataset['SentimentText']
    print("len of sentences: ", len(sentences))
    # removing non-ascii charactars
    print("Removing non-ascii charachars...")
    sentences_ascii=[]
    for sentence in sentences:
        sentences_ascii.append(''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in sentence]))
#    sentences = sentences.apply( lambda x:  unidecode(unicode(x, encoding = "utf-8")))
    print("Getting words...")
    words = list(map(lambda x: get_filtered_tokens(x), sentences_ascii))
    print("word looks like: ", words[0:2])
    documents = list(zip(words, categories))
    print("document looks like: ", documents[0:2])
    print("Document created!")
    all_words = []
    for words_in_sentences in words:
        for w in words_in_sentences:
            all_words.append(w.lower())
    pickle.dump(documents,open("documents.pkl","wb"))
    pickle.dump(all_words,open("all_words.pkl","wb"))
    return documents, all_words


def get_classification_data(documents, all_words, reload=True):
    if reload:
        training_set = pickle.load(open("training_set.pkl", "rb"))
        testing_set = pickle.load(open("testing_set.pkl", "rb"))
        return training_set, testing_set
    random.shuffle(documents)
#    print("document looks like: ", documents[0:2])

    #    words_dist = nltk.FreqDist(all_words).most_common(100)
#    print("len(all_words): ", len(all_words))

    featuresets = [(find_features(sentence, all_words), category) for (sentence, category) in documents]
#    print("len of featuresets: ", len(featuresets))
#    print("featuresets looks like: ", featuresets[0])
    train_set_size = int(len(featuresets)/10*9)
    training_set = featuresets[:train_set_size]
    testing_set = featuresets[train_set_size:]
    pickle.dump(training_set, open("training_set.pkl", "wb"))
    pickle.dump(testing_set, open("testing_set.pkl", "wb"))
    return training_set, testing_set

def text_classifier():
    #    documents, all_words = movie_reviews_documents()
    documents, all_words = sentiment_analysis_dataset_documents()
    print("got documents")
    training_set, testing_set = get_classification_data(documents, all_words)
    print("got training and testing set")
#    print("train_set_size[0]: ",training_set[0])
#    X,
#    SVC_classifier = GridSearchCV(SVC(), cv=5, param_grid={})
#    SVC_classifier.fit(featuresets, )
#    print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
#    text_classifier.classifier = BNB_classifier

    naiveBayesClassifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Naive Bayes Classifier accuracy percent:",(nltk.classify.accuracy(naiveBayesClassifier, testing_set))*100)

    training_set, testing_set = get_classification_data(documents, all_words)
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set)*100)

    training_set, testing_set = get_classification_data(documents, all_words)
    BNB_classifier = SklearnClassifier(BernoulliNB())
    BNB_classifier.train(training_set)
    print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set)*100)

    training_set, testing_set = get_classification_data(documents, all_words)
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

    training_set, testing_set = get_classification_data(documents, all_words)
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

    training_set, testing_set = get_classification_data(documents, all_words)
    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

    training_set, testing_set = get_classification_data(documents, all_words)
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

    training_set, testing_set = get_classification_data(documents, all_words)
    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training_set)
    print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

    voted_classifier = VoteClassifier(naiveBayesClassifier,
                                      NuSVC_classifier,
                                      LinearSVC_classifier,
                                      SGDClassifier_classifier,
                                      MNB_classifier,
                                      BNB_classifier,
                                      LogisticRegression_classifier)

    print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

    print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
    print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)

    return voted_classifier

def train_classifier(reload=True):
    if not reload:
        classifier = text_classifier()
        pickle.dump(classifier, open("classifier.pkl", "wb"))
        return classifier
    else:
        return pickle.load(open("classifier.pkl", "rb"))

def get_verb_tags():
    return ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]


def list_pos_tagger(all_sentences):
    for sentence in all_sentences:
        pos_tagger(sentence)


def get_filtered_tokens(text):
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
#    print("shape, type(filtered_tokens): ", np.shape(filtered_tokens), type(filtered_tokens))

#    print("np.shape(filtered_tokens): ", np.shape(filtered_tokens))
#    print "most common words: ", nltk.FreqDist(filtered_tokens).most_common(20)
    return filtered_tokens

def classify_tweet(text, classifier):
    tokens = get_filtered_tokens(text)
    cl = classifier.classify(find_features(text, tokens))
    cnf = classifier.confidence(find_features(text, tokens))
    print("polarity: ", cl, ", confidency: ", cnf)

def classify_tweet_nltk(text, sid):
    score = sid.polarity_scores(text)
    print("nltk score: ", score)

def pos_tagger(text):

    tokens = get_filtered_tokens(text)
    tagged = pos_tag(tokens)
    sid = SentimentIntensityAnalyzer()

    classify_tweet(text, sid)
#    print "tagged: ", tagged

    verbs = [w[0] for w in tagged if w[1] in get_verb_tags()]
#    print "verbs: ", verbs


def sentence_tokenize(text):
    return nltk.sent_tokenize(text)

def get_tweet_text():
    # TODO: is buggy!
    f = open("Archive/results/filtered_tweets/part-00000") # 36726-7 tweets in this file
#    reg = re.compile("\w+:[[\[.+\]]|[^,]*]*[,\n]")
#    keys = "[\bid\b | \bcreated_at\b | \btext\b | \blang\b | \bsource\b | \buser\b | \bentities\b | \bretweeted\b | \bretweet_count\b | \n]"
#    keys = "[id:|created_at:|text:|lang:|source:|user:|entities:|retweeted:|retweet_count:|\n]"
#    reg = re.compile(keys+"[^"+keys+"]*")
#    get_tweet_text.structured_tweets = map(lambda x: x[0:-1].split(':',1), reg.findall(f.read()))
    reg = re.compile("text:.*?lang")
    get_tweet_text.structured_tweets = map(lambda x: re.sub(r'[^\x00-\x7F]+',' ', x[5:-6]), reg.findall(f.read()))
#    print len(get_tweet_text.structured_tweets)
    f.close()
    return get_tweet_text.structured_tweets

if __name__ == "__main__":

    #sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good. He was going to university."""
    #f = open("Archive/results/filtered_tweets/part-00000")
    vclassifier = train_classifier()
    all_sentences = []
    for tweet in get_tweet_text():
        all_sentences.append(tweet)
    for sentence in sentences:
        print(sentence)
        classify_tweet(sentence, vclassifier)
        classify_tweet_nltk(sentence, SentimentIntensityAnalyzer())
