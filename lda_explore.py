
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from ast import literal_eval

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt


# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[2]:


THEMES = [5, 6, 26, 33, 139, 163, 232, 313, 339, 350, 406, 409, 555, 589,
          597, 634, 660, 695, 729, 766, 773, 793, 800, 810, 852, 895, 951, 975]
TRAIN_DATA_PATH = '../train.csv'
TEST_DATA_PATH = '../test.csv'
VALIDATION_DATA_PATH = '../validation.csv'


# In[3]:


def get_data(path, preds=None, key=None):
    data = pd.read_csv(path)
    data = data.rename(columns={ 'pages': 'page'})
    data.body = data.body.str.strip('{}"')
    data = groupby_process(data)
    data.themes = data.themes.apply(lambda x: literal_eval(x))
    return data


# In[4]:


def groupby_process(df):
    new_df = df.sort_values(['process_id', 'page'])
    new_df = new_df.groupby(
                ['process_id', 'themes'],
                group_keys=False
            ).apply(lambda x: x.body.str.cat(sep=' ')).reset_index()
    new_df = new_df.rename(index=str, columns={0: "body"})
    return new_df


# In[5]:


train_data = get_data(TRAIN_DATA_PATH)
test_data = get_data(TEST_DATA_PATH)
validation_data = get_data(VALIDATION_DATA_PATH)


# In[6]:


train_data.themes = train_data.themes.apply(lambda x: list(set(sorted([i if i in THEMES else 0 for i in x]))))
test_data.themes = test_data.themes.apply(lambda x: list(set(sorted([i if i in THEMES else 0 for i in x]))))
validation_data.themes = validation_data.themes.apply(lambda x: list(set(sorted([i if i in THEMES else 0 for i in x]))))


# In[7]:


len(train_data), len(validation_data), len(test_data)


# In[8]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield([x for x in sentence.split(" ") if len(x) > 1])

train_words = list(sent_to_words(train_data.body.tolist()))


print(train_words[0][:20])


# In[9]:


len(train_words)


# In[10]:


# # Build the bigram and trigram models
# bigram = gensim.models.Phrases(train_words, min_count=50, threshold=100) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[train_words], min_count=5, threshold=100)  

# # Faster way to get a sentence clubbed as a trigram/bigram
# bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)

# # See trigram example
# print(trigram_mod[bigram_mod[train_words[0]]])


# In[11]:


# def make_bigrams(texts):
#     return [bigram_mod[doc] for doc in texts]

# def make_trigrams(texts):
#     return [trigram_mod[bigram_mod[doc]] for doc in texts]


# In[12]:


# train_words = make_bigrams(train_words)
# train_words = make_trigrams(train_words)


# In[25]:


# Create Dictionary
#id2word = corpora.Dictionary(train_words)

id2word = corpora.Dictionary.load("dicts/big_dict")

# In[26]:


def save_dic(dic, filename="dic"):
    with open(filename, "wb") as handle:
        dic.save(handle)


# In[27]:


#save_dic(id2word, "dicts/big_dict")


# In[28]:


#len(id2word.dfs), id2word.dfs


# In[40]:


#from copy import deepcopy

#copy_dict = deepcopy(id2word)
#copy_dict.filter_extremes(no_below=50, no_above=.5, keep_n=None)


# In[41]:


#len(copy_dict.dfs), copy_dict.dfs


# In[42]:


#id2word = deepcopy(copy_dict)
#del(copy_dict)


# In[43]:


#[(id2word[x], y) for (x, y) in sorted(id2word.dfs.items(), key=lambda x: x[1], reverse=True)]


# In[44]:


#save_dic(id2word, "big_dict")


# In[ ]:


train_corpus = [id2word.doc2bow(text) for text in train_words]
del(train_data)
del(validation_data)
del(test_data)
del(train_words)

# In[ ]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus,
                                           id2word=id2word,
                                           num_topics=50, 
                                           random_state=42,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[ ]:


pprint(lda_model.print_topics())


# In[ ]:


lda_model.save("models/lda_big_50")


# In[ ]:


# Visualize the topics
#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(lda_model, train_corpus, id2word)
#vis


# In[ ]:


#import os
#os.environ.update({'MALLET_HOME':"/home/isis/Davi_Alves/data/parts/topic_modeling/mallet/mallet-2.0.8"})

#mallet_path = "/home/isis/Davi_Alves/data/parts/topic_modeling/mallet/mallet-2.0.8/bin/mallet" # update this path

#ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=train_corpus, num_topics=10, id2word=id2word, workers=10)


# In[ ]:


#pprint(ldamallet.print_topics())


# In[ ]:


#with open("models/ldamallet_big", "wb") as handle:
#    ldamallet.save(handle)

