#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import pickle
import pandas as pd
import numpy as np
from joblib import dump, load
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss, zero_one_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from xgboost.sklearn import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uni

from sklearn.externals import joblib

from sklearn.model_selection import RandomizedSearchCV


# In[2]:


THEMES = [5, 6, 26, 33, 139, 163, 232, 313, 339, 350, 406, 409, 555, 589,
          597, 634, 660, 695, 729, 766, 773, 793, 800, 810, 852, 895, 951, 975]
TRAIN_DATA_PATH = 'jurix-2020-pedro/csv/train_small.csv'
TEST_DATA_PATH = 'jurix-2020-pedro/csv/test_small.csv'
VALIDATION_DATA_PATH = 'jurix-2020-pedro/csv/validation_small.csv'


# In[3]:


def groupby_process(df):
    new_df = df.sort_values(['process_id', 'page'])
    new_df = new_df.groupby(
                ['process_id', 'themes'],
                group_keys=False
            ).apply(lambda x: x.body.str.cat(sep=' ')).reset_index()
    new_df = new_df.rename(index=str, columns={0: "body"})
    return new_df

def get_data(path, preds=None, key=None):
    data = pd.read_csv(path)
    data = data.rename(columns={ 'pages': 'page'})
#     data["preds"] = preds[key]
#     data = data[data["preds"] != "outros"]
    data = groupby_process(data)
    data.themes = data.themes.apply(lambda x: literal_eval(x))
    return data

def transform_y(train_labels, test_labels):
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels)

    mlb_train = mlb.transform(train_labels)
    mlb_test = mlb.transform(test_labels)

    print(mlb.classes_)

    return mlb_train, mlb_test, mlb


# In[4]:


train_data = get_data(TRAIN_DATA_PATH)
test_data = get_data(TEST_DATA_PATH)
validation_data = get_data(VALIDATION_DATA_PATH)

train_data.themes = train_data.themes.apply(lambda x: list(set(sorted([i if i in THEMES else 0 for i in x]))))
test_data.themes = test_data.themes.apply(lambda x: list(set(sorted([i if i in THEMES else 0 for i in x]))))
validation_data.themes = validation_data.themes.apply(lambda x: list(set(sorted([i if i in THEMES else 0 for i in x]))))

y_train, y_test, mlb = transform_y(train_data.themes, test_data.themes)

X_train = train_data.body
X_test = test_data.body
print('X_train: {}, \n\ty_train: {}'.format(X_train.shape, y_train.shape))
print('X_test: {}, \n\ty_test: {}'.format(X_test.shape, y_test.shape))
print('Classes: ', mlb.classes_)
print('We\'re classifying {} themes!'.format(y_train.shape[1]))


# In[5]:


vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True,
                                   min_df=50, max_df=0.5)

xgboost = OneVsRestClassifier(XGBClassifier(
                n_jobs=-1,
                max_depth=4,
                learning_rate=0.1,
                n_estimators=500,
            ),
            n_jobs=1)


# In[6]:


# from scipy.stats import randint as sp_randint
# from scipy.stats import uniform as sp_uni

param_dist = {"estimator__max_depth": sp_randint(1, 8),
              "estimator__learning_rate": [0.1, 0.3, 0.5],
              "estimator__n_estimators": [30, 100, 300, 500, 1000]}


# In[7]:


X_train = vectorizer.fit_transform(X_train)
X_valid = vectorizer.transform(validation_data.body)
X_test = vectorizer.transform(X_test)
y_valid = mlb.transform(validation_data.themes)


# In[9]:


# len(vectorizerctorizer.vocabulary_)


# In[10]:


# from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(xgboost, param_distributions=param_dist,
                                   n_iter=20, n_jobs=1, iid=False, refit=False,
                                   verbose=2, random_state=42)
random_search.fit(X_valid, y_valid)


# In[ ]:


random_search.cv_results_


# In[ ]:


best_params = random_search.best_params_; best_params


# In[ ]:


xgboost = random_search.estimator.set_params(**best_params); xgboost


# In[ ]:


xgboost.fit(X_train, y_train)


# In[ ]:


target_names=[str(x) for x in mlb.classes_]


# In[ ]:


preds_test = xgboost.predict(X_test)
print(classification_report(y_test, preds_test, target_names=target_names, digits=4))
print(accuracy_score(y_test, preds_test))


# In[ ]:


# from sklearn.externals import joblib

joblib.dump(xgboost, './models/tfidf_xgboot.pkl')
joblib.dump(vectorizer, "./models/tfidf_vectorizer.pkl")

