#!/usr/bin/env python
# coding: utf-8

# #  KoNLPy를 사용한 영화 리뷰 분석

# MeCab을 설치하려면 다음 명령을 실행하세요.
# 
# `$ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)`
# 
# 파이썬 가상 환경에서 MeCab을 설치하려면 이 깃허브에 포함된 `mecab.sh` 파일을 실행하세요.
# 
# 최신 macOS Mojave에서는 Mecab에 필요한 jpype 라이브러리가 컴파일 오류가 발생할 수 있습니다. 이런 경우 다음 명령으로 konlpy를 설치해 주세요.
# 
# `$ export MACOSX_DEPLOYMENT_TARGET=10.10 CFLAGS='-stdlib=libc++' pip install konlpy`
# 
# <b><font color='red'>주의: 코랩에서 실행하는 경우 아래 셀을 실행하고 ⌘+M . 또는 Ctrl+M . 을 눌러 런타임을 재시작한 다음 처음부터 다시 실행해 주세요.</font></b>

# In[1]:


get_ipython().system('bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)')


# In[2]:


# 노트북이 코랩에서 실행 중인지 체크합니다.
import os
import sys
if 'google.colab' in sys.modules:
    # 사이킷런 최신 버전을 설치합니다.
    get_ipython().system('pip install -q --upgrade scikit-learn')
    # 데이터를 다운받고 압축을 풉니다.
    get_ipython().system('wget -q -O data.tar.gz https://bit.ly/data-tar-gz')
    get_ipython().system('tar -xzf data.tar.gz')


# 최신 `tweepy`를 설치할 경우 `StreamListener`가 없다는 에러가 발생(https://github.com/tweepy/tweepy/issues/1531) 하므로 3.10버전을 설치해 주세요.

# In[3]:


get_ipython().system('pip install -q tweepy==3.10')

try:
    import konlpy
except:
    get_ipython().system('pip install -q konlpy')
    import konlpy

import pandas as pd
import numpy as np

konlpy.__version__


# 데이터 파일을 읽어 리뷰 텍스트와 점수를 text_train, y_train 변수에 저장합니다. 데이터 파일의 내용은 번호, 텍스트, 레이블이 탭으로 구분되어 한 라인에 한개의 데이터 샘플이 들어 있습니다.

# In[4]:


df_train = pd.read_csv('data/ratings_train.txt', delimiter='\t', keep_default_na=False)

df_train.head()


# In[5]:


text_train, y_train = df_train['document'].values, df_train['label'].values


# 같은 방식으로 테스트 데이터를 읽습니다.

# In[6]:


df_test = pd.read_csv('data/ratings_test.txt', delimiter='\t', keep_default_na=False)
text_test = df_test['document'].values
y_test = df_test['label'].values


# 훈련 데이터와 테스트 데이터의 크기를 확인합니다.

# In[7]:


len(text_train), np.bincount(y_train)


# In[8]:


len(text_test), np.bincount(y_test)


# #### Okt

# KoNLPy 0.4.5 버전부터 `Twitter` 클래스가 `Okt` 클래스로 바뀌었습니다. [open-korean-text](https://github.com/open-korean-text/open-korean-text) 프로젝트는 [twitter-korean-text](https://github.com/twitter/twitter-korean-text) 프로젝트의 공식 포크입니다.
# 
# 피클링을 위해 \_\_setstate__, \_\_getstate__ 메서드를 추가하여 Okt 클래스를 감쌉니다.

# In[9]:


from konlpy.tag import Okt

class PicklableOkt(Okt):

    def __init__(self, *args):
        self.args = args
        Okt.__init__(self, *args)
    
    def __setstate__(self, state):
        self.__init__(*state['args'])

    def __getstate__(self):
        return {'args': self.args}

okt = PicklableOkt()


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

param_grid = {'tfidfvectorizer__min_df': [3, 5, 7],
              'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'logisticregression__C': [0.1, 1, 10]}
pipe = make_pipeline(TfidfVectorizer(tokenizer=okt.morphs),
                     LogisticRegression())
grid = GridSearchCV(pipe, param_grid, n_jobs=-1)

# 그리드 서치를 수행합니다
grid.fit(text_train[:1000], y_train[:1000])
print("최상의 크로스 밸리데이션 점수: {:.3f}".format(grid.best_score_))
print("최적의 크로스 밸리데이션 파라미터: ", grid.best_params_)


# In[11]:


tfidfvectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
X_test = tfidfvectorizer.transform(text_test[:1000])
logisticregression = grid.best_estimator_.named_steps["logisticregression"]
score = logisticregression.score(X_test, y_test[:1000])

print("테스트 세트 점수: {:.3f}".format(score))


# #### Mecab

# In[12]:


from konlpy.tag import Mecab

class PicklableMecab(Mecab):

    def __init__(self, *args):
        self.args = args
        Mecab.__init__(self, *args)
    
    def __setstate__(self, state):
        self.__init__(*state['args'])

    def __getstate__(self):
        return {'args': self.args}

mecab = PicklableMecab()


# In[13]:


pipe = make_pipeline(TfidfVectorizer(tokenizer=mecab.morphs), LogisticRegression())
grid = GridSearchCV(pipe, param_grid, n_jobs=-1)

# 그리드 서치를 수행합니다
grid.fit(text_train[:1000], y_train[:1000])
print("최상의 크로스 밸리데이션 점수: {:.3f}".format(grid.best_score_))
print("최적의 크로스 밸리데이션 파라미터: ", grid.best_params_)


# In[14]:


tfidfvectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
X_test = tfidfvectorizer.transform(text_test[:1000])
logisticregression = grid.best_estimator_.named_steps["logisticregression"]
score = logisticregression.score(X_test, y_test[:1000])

print("테스트 세트 점수: {:.3f}".format(score))

