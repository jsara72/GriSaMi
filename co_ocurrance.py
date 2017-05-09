from sentiment_analysis import get_tweet_text as get_tweet_text
from sentiment_analysis import get_filtered_tokens as get_filtered_tokens
from sklearn.decomposition import PCA
from sklearn import manifold
import numpy as np
import nltk
import argparse
import pickle
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

def get_filtered_words(tweet):
    word_list = []
    tokens = get_filtered_tokens(tweet)
    for token in tokens:
        if len(token) > 3:# and token.lower() not in ["https", "march", "women", "womensmarch"]:
            word_list.append(token.lower())
    return word_list

def save_plot(labels, X_true):
    fig = plt.figure()
    ax = plt.axes([0., 0., 1., 1.])

    plt.scatter(X_true[:, 0], X_true[:, 1], color='navy')
    for label, x, y in zip(labels, X_true[:, 0], X_true[:, 1]):
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

    else:
        all_words = []
        counter = 0
        for tweet in get_tweet_text():
            counter += 1
            if counter % 10000 == 0:
                print("in line ", counter)
            all_words.extend(get_filtered_words((tweet)))
        print("shape, type(all_words): ", np.shape(all_words), type(all_words))
        vocab_size = 200
        co_oc = np.zeros((vocab_size, vocab_size))
        #    print("all words: ", all_words[0:2])
        most_common_words_freq = nltk.FreqDist(all_words).most_common(vocab_size)
        most_common_words = [word[0] for word in most_common_words_freq]
        word_to_id = {word: id for (word, id) in map(lambda ind: (most_common_words[ind], ind), range(len(most_common_words)))}
        #    print(word_to_id)
        for tweet in get_tweet_text():
            words = [w for w in get_filtered_words(tweet) if w in most_common_words]
            for first_word in words:
                for second_word in words:
                    co_oc[word_to_id[first_word]][word_to_id[second_word]] += 1

        pickle.dump(co_oc,open("co_oc.pkl", "wb"))
        pickle.dump(most_common_words,open("co_oc_most_common_words.pkl", "wb"))
    return co_oc, most_common_words

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--reload', action="store_true",
                    help='reloads last saved objects to run faster')
    args = parser.parse_args()

#    print(co_oc)
    co_oc, most_common_words = get_co_oc(args.reload)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=0,
                   dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(co_oc).embedding_

    pca = PCA(n_components=10)
    word_emb = pca.fit_transform(co_oc)
    print (word_emb)
    print (np.shape(word_emb[:, 1:3]))
    save_plot(most_common_words, word_emb[:, 1:3])
#    save_plot(most_common_words, pos)
    print("pca variance ratio: ", pca.explained_variance_ratio_)


