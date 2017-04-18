
# coding: utf-8

# In[1]:

import json
import pandas as pd
from pandas.io.json import json_normalize
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


# In[19]:

def griffin():
    
    order = ['id','time','text','language','retweeted','retweet counts','']
    
    idt = re.compile("id:.*created_at:")
    idt_temp = list(map(lambda x: re.sub(r'[^\x00-\x7F]+',' ', x[3:-13]), idt.findall(f.read())))
    
    f.seek(0)
    time = re.compile("created_at:.*text:")
    time_temp = list(map(lambda x: re.sub(r'[^\x00-\x7F]+',' ', x[11:-7]), time.findall(f.read())))
    
    f.seek(0)

    text = re.compile("text:.*?lang:",re.DOTALL)
    text_temp = list(map(lambda x: re.sub(r'[^\x00-\x7F]+',' ', x[5:-7]), text.findall(f.read())))
    
    
    f.seek(0)
    length = re.compile("lang:.*source:")
    length_temp = list(map(lambda x: re.sub(r'[^\x00-\x7F]+',' ', x[5:-9]), length.findall(f.read())))
    
    f.seek(0)
    retweeted = re.compile("retweeted:.*retweet_count:")
    retweeted_temp = list(map(lambda x: re.sub(r'[^\x00-\x7F]+',' ', x[10:-16]), retweeted.findall(f.read())))
    
    f.seek(0)
    retweet_counts = re.compile("retweet_counts:.*")
    retweet_counts_temp = list(map(lambda x: re.sub(r'[^\x00-\x7F]+',' ', x[14:-1]), retweet_counts.findall(f.read())))
    
    #print(len(idt_temp),len(time_temp),len(text_temp),len(length_temp),len(retweeted_temp),len(retweet_counts_temp))
    
    df = pd.DataFrame({order[0]: idt_temp,
                        order[1]: time_temp,
                        order[2]: text_temp,
                        order[3]: length_temp,
                       order[4]: retweeted_temp})
    #                  order[5]: retweet_counts_temp})
    return df


# In[20]:

f = open("part-00000") # 36726-7 tweets in this file


# In[21]:

df = griffin()


# In[26]:

df['language'].values


# In[ ]:



