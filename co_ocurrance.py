from sentiment_analysis import get_tweet_text as get_tweet_text
from sentiment_analysis import get_filtered_tokens as get_filtered_tokens
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.cluster import KMeans
import numpy as np
import nltk
import argparse
import pickle
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import rainbow
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer



def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return wn.NOUN


def get_filtered_words(tweet, remove_outliers=False):
    word_list = []
    tag_list = []
    tweet = tweet.lower()
    tokens = get_filtered_tokens(tweet)
    tags = nltk.pos_tag(tokens)
    for i in range(len(tokens)):
        token = tokens[i].lower()
        tag = tags[i]
        wn_tag = penn_to_wn(tag[1])
#        if remove_outliers and WordNetLemmatizer().lemmatize(tag[0],wn_tag) in ["http", "march", "woman", "womensmarch", "today"]:
#            continue
        if len(token) > 3 and not token.startswith("//"):
            tag_list.append(tag[1])
            word_list.append(WordNetLemmatizer().lemmatize(tag[0],wn_tag))
    return word_list, tag_list

def save_plot(labels, X_true, cluster_labels, n_clusters):
    fig = plt.figure()
    ax = plt.axes([0., 0., 1., 1.])
    chosen_colors = rainbow(np.linspace(0, 1, n_clusters))
#    all_colors = ['red', 'green', 'blue', 'yellow', 'black']
#    chosen_colors = all_colors[:n_clusters]
    print("len(X_true)", len(X_true))
    print("len(labels)", len(labels))
    
#    print("cluster labels look like: ", cluster_labels[0:10])
    for c in range(n_clusters):
        print("for cluster ", c)
        X_Y = [X_true[i] for i in range(len(X_true)) if cluster_labels[i] == c]
        labels_c = [labels[i] for i in range(len(labels)) if cluster_labels[i] == c]
        print(labels_c)
#        X_Y_indices = [i for i in range(len(X_true)) if cluster_labels[i] == c]
#        print("X_Y_indices: ", np.shape(X_Y_indices), type(X_Y_indices))
        print("X_Y: ", np.shape(X_Y), type(X_Y))
        X = [f for f,s in X_Y]
        Y = [s for f,s in X_Y]
        print("This class size:", len(X))
        plt.scatter(X, Y, color=chosen_colors[c])
        for label, x, y, cluster_label in zip(labels, X_true[:, 0], X_true[:, 1], cluster_labels):
            plt.annotate(
                 label,
                 xy=(x, y), xytext=(-20, 20),
                 textcoords='offset points', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

#    plt.legend(scatterpoints=1, loc='best', shadow=False)
#
#    similarities = similarities.max() / similarities * 100
#    similarities[np.isinf(similarities)] = 0
#
#    # Plot the edges
#    start_idx, end_idx = np.where(pos)
#    # a sequence of (*line0*, *line1*, *line2*), where::
#    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
#    segments = [[X_true[i, :], X_true[j, :]]
#                for i in range(len(pos)) for j in range(len(pos))]
#    values = np.abs(similarities)
#    lc = LineCollection(segments,
#                        zorder=0, cmap=plt.cm.Blues,
#                        norm=plt.Normalize(0, values.max()))
#    lc.set_array(similarities.flatten())
#    lc.set_linewidths(0.5 * np.ones(len(segments)))
#    ax.add_collection(lc)

    plt.show()

def get_co_oc(reload):
    if reload:
        co_oc = pickle.load(open("co_oc.pkl", "rb"))
        most_common_words = pickle.load(open("co_oc_most_common_words.pkl", "rb"))
        return co_oc, most_common_words
    else:
        all_words = []
        counter = 0
        for tweet in get_tweet_text():
            counter += 1
            if counter % 10000 == 0:
                print("in line ", counter)
            words, tages = get_filtered_words(tweet)
            all_words.extend(words)
        print("shape, type(all_words): ", np.shape(all_words), type(all_words))
        vocab_size = 200
        co_oc = np.zeros((vocab_size, vocab_size))
        #    print("all words: ", all_words[0:2])
        most_common_words_freq = nltk.FreqDist(all_words).most_common(vocab_size)
        most_common_words = [word[0] for word in most_common_words_freq]
        word_to_id = {word: id for (word, id) in map(lambda ind: (most_common_words[ind], ind), range(len(most_common_words)))}
        #    print(word_to_id)
        for tweet in get_tweet_text():
            ws, ts = get_filtered_words(tweet)
            words = []
            for i in range(len(ws)):
                if ws[i] in most_common_words:# and is_noun(ts[i]):
                    words.append(ws[i])
#            words = [w for w, t in get_filtered_words(tweet) if w in most_common_words and is_noun(t)]
            for first_word in words:
                for second_word in words:
                    if first_word is not second_word:
                        co_oc[word_to_id[first_word]][word_to_id[second_word]] += 1
        co_oc_without_outlier = []
        most_common_words_without_outlier = []
        outlier_ids = [word_to_id[outlier] for outlier in ["http", "march", "woman", "womensmarch", "today"]]
        print("outlier_ids: ", outlier_ids)
        print("most common words size: ", len(most_common_words))
        print("co_oc size: ", len(co_oc))
        for i in range(len(co_oc)):
#            print(np.shape(co_oc[i]))
            if i not in outlier_ids:
                co_oc_without_outlier.append(co_oc[i])
                most_common_words_without_outlier.append(most_common_words[i])
        print("co_oc_without_outlier: ", np.shape(co_oc_without_outlier))
        pickle.dump(co_oc_without_outlier,open("co_oc.pkl", "wb"))
        pickle.dump(most_common_words_without_outlier, open("co_oc_most_common_words.pkl", "wb"))
        return co_oc_without_outlier, most_common_words_without_outlier

def do_clustering(co_oc):

    loss_values = []
    labels = []
    chosen_n_clusters = 6
    for n_clusters in range(2,50):
        clusters = KMeans(n_clusters=n_clusters, random_state=0, n_init=10, init="k-means++")
        clusters.fit(co_oc)
        loss_values.append(clusters.inertia_)
        if n_clusters == chosen_n_clusters:
            labels = clusters.labels_
#    print("clustering loss values for n_clusters in range(2,20): ", loss_values)
    fig = plt.figure()
    plt.plot(range(2,50), loss_values)
    plt.xlabel("Number of clusters")
    plt.ylabel("K-Means loss value")
    plt.show()
    return labels, chosen_n_clusters

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute', action="store_true",
                    help='recomputes everyting doesn\'t reload last saved objects')
    args = parser.parse_args()
#    print(co_oc)
    co_oc, most_common_words = get_co_oc(not args.recompute)
    labels, n_clusters = do_clustering(co_oc)
#    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=0,
#                   dissimilarity="precomputed", n_jobs=1)
#    pos = mds.fit(co_oc).embedding_

    pca = PCA(n_components=2)
    word_emb = pca.fit_transform(co_oc)
#    print (word_emb)
#    print (np.shape(word_emb[:, 1:3]))
    save_plot(most_common_words, word_emb, labels, n_clusters)
    print("pca variance ratio: ", pca.explained_variance_ratio_)

#    save_plot(most_common_words, pos)


