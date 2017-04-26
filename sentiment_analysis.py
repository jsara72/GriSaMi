# author: jsara72
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import random
import nltk
from nltk.corpus import movie_reviews
import pandas as pd
import re
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import csv
from io import StringIO
#from unidecode import unidecode


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
        
    print "movie_reviews.words(fileid): ", movie_reviews.fileids(category)[:2]
    print "movie_reviews.categories(): ", movie_reviews.categories()
    return documents, movie_reviews.words


def sentiment_analysis_dataset_documents():
    print "Reading dataset..."
    dataset = pd.read_csv("Sentiment Analysis Dataset.csv", quotechar='"', quoting=0,  error_bad_lines=False)
    categories = dataset['Sentiment']
    sentences = dataset['SentimentText']
    # removing non-ascii charactars
    print "Removing non-ascii charachars..."
    sentences = sentences.apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
#    sentences = sentences.apply( lambda x:  unidecode(unicode(x, encoding = "utf-8")))
    print "Getting words..."
    words = map(lambda x: get_filtered_tokens(x), sentences.values)
    print "word looks like: ", words[0:2]
    documents = zip(words, categories)
    print "Document created!"
    return documents, words


def text_classifier():
#    documents = movie_reviews_documents()
    documents, words = sentiment_analysis_dataset_documents()
    random.shuffle(documents)
    
    all_words = []
    for w in words():
        all_words.append(w.lower())
    all_words = nltk.FreqDist(all_words)
    print "len(all_words): ", len(all_words)
    print "len(list(all_words.keys())): ", len(list(all_words.keys()))

    word_features = list(all_words.keys())[:3000]
#    print "most and least word features: ", word_features[0:10], word_features[-10:-1]


    featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]
    print "len of featuresets: ", len(featuresets)

    training_set = featuresets[:1900]
    testing_set = featuresets[1900:]

#    SVC_classifier = SklearnClassifier(SVC())
#    SVC_classifier.train(training_set)
#    print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
#    MNB_classifier = SklearnClassifier(MultinomialNB())
#    MNB_classifier.train(training_set)
#    print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set))

    BNB_classifier = SklearnClassifier(BernoulliNB())
    BNB_classifier.train(training_set)
    print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set))
    text_classifier.classifier = BNB_classifier


def get_verb_tags():
    return ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]


def list_pos_tagger(all_sentences):
    for sentence in all_sentences:
        pos_tagger(sentence)


def get_filtered_tokens(text):
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
#    print "most common words: ", nltk.FreqDist(filtered_tokens).most_common(20)
    return filtered_tokens

def classify_tweet(text):
    tokens = get_filtered_tokens(text)
    cl = text_classifier.classifier.classify(find_features(text, tokens))
    print text, cl

def pos_tagger(text):

    tokens = get_filtered_tokens(text)
    tagged = pos_tag(tokens)
    classify_tweet(text)
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
    return get_tweet_text.structured_tweets


#sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good. He was going to university."""
#f = open("Archive/results/filtered_tweets/part-00000")
text_classifier()
all_sentences = []
for tweet in get_tweet_text():
    all_sentences.append(tweet)
list_pos_tagger(all_sentences)
