#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 노트북이 코랩에서 실행 중인지 체크합니다.
import os
import sys
if 'google.colab' in sys.modules:
    # 사이킷런 최신 버전을 설치합니다.
    get_ipython().system('pip install -q --upgrade scikit-learn')


# ### 나만의 추정기 만들기

# In[2]:


from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, first_paramter=1, second_parameter=2):
        # __init__ 메소드에 필요한 모든 매개변수를 나열합니다
        self.first_paramter = 1
        self.second_parameter = 2
        
    def fit(self, X, y=None):
        # fit 메소드는 X와 y 매개변수만을 갖습니다
        # 비지도 학습 모델이더라도 y 매개변수를 받도록 해야합니다!
        
        # 모델 학습 시작
        print("모델 학습을 시작합니다")
        # 객체 자신인 self를 반환합니다
        return self
    
    def transform(self, X):
        # transform 메소드는 X 매개변수만을 받습니다
        
        # X를 변환합니다
        X_transformed = X + 1
        return X_transformed

