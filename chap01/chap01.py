#!/usr/bin/env python
# coding: utf-8

# ### Jupyter Notebook

# In[1]:


pip install numpy scipy matplotlib ipython scikit-learn pandas pillow imageio


# In[2]:


pip install mglearn


# ### Numpy

# In[3]:


import numpy as np


# In[4]:


x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n", x)


# ### SciPy

# In[5]:


from scipy import sparse

# 대각선 원소는 1이고 나머지는 0인 2차원 Numpy 배열을 만듭니다.
eye = np.eye(4)
print("NumPy 배열:\n", eye)


# In[6]:


# Numpy 배열을 CSR 포맷의 SciPy 희박 행렬로 변환합니다.
# 0이 아닌 원소만 저장됩니다.
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy의 CSR 행렬:\n", sparse_matrix)


# In[7]:


data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO 표현:\n", eye_coo)


# ### matplotilib

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# -10에서 10Rkwl 100개의 간격으로 나뉘어진 배열을 생성합니다.
x = np.linspace(-10, 10, 100)
# 사인 함수를 사용하여 y 배열을 생성합니다.
y = np.sin(x)
# plot 함수는 한 배열의 값을 다른 배열에 대응해서 선 그래프를 그립니다.
plt.plot(x, y, marker='x')


# ### pandas

# In[9]:


import pandas as pd

# 회원 정보가 들어간 간단한 데이터셋을 생성합니다.
data = {'Name' : ["John", "Anna", "Peter", "Linda"],
       'Location' : ["New York", "Paris", "Berlin", "London"],
        'Age' : [24, 13, 53, 33]
       }
data_pandas = pd.DataFrame(data)
# 주피터 노트북은 Dataframe을 미려하게 출력해줍니다.
data_pandas


# In[10]:


# Age 열의 값이 30 이상인 모든 행을 선택합니다.
data_pandas[data_pandas.Age > 30]


# ### mglearn

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn


# ### version

# In[12]:


import sys
print("Python version:", sys.version)

import pandas as pd
print("pandas version:", pd.__version__)

import matplotlib
print('matplotlib version:', matplotlib.__version__)

import numpy as np
print("Numpy version:", np.__version__)

import scipy as sp
print("SciPy version:", sp.__version__)

import IPython
print("IPython version:", IPython.__version__)

import sklearn
print("scikit-learn version:", sklearn.__version__)


# # 첫 번째 애플리케이션 : 붓꽃의 품종 분류

# ### 데이터 적재

# In[13]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[14]:


print("iris_dataset의 키:\n", iris_dataset.keys())


# In[15]:


print(iris_dataset['DESCR'][:193] + "\n...")


# In[16]:


print("타깃의 이름:", iris_dataset['target_names'])


# In[17]:


print("특성의 이름:\n", iris_dataset['feature_names'])


# In[18]:


print("data의 타입:", type(iris_dataset['data']))


# In[19]:


print("data의 크기:", iris_dataset['data'].shape)


# In[20]:


print("data의 처음 다섯 행:\n", iris_dataset['data'][:5])


# In[21]:


print("target의 타입:", type(iris_dataset['target']))


# In[22]:


print("target의 크기:", iris_dataset['target'].shape)


# In[23]:


print("타깃:\n", iris_dataset['target'])


# ### 성과 측정

# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[25]:


print("X_train 크기 : ", X_train.shape)
print("y_train 크기 : ", y_train.shape)


# In[26]:


print("X_test 크기 : ", X_test.shape)
print("y_test 크기 : ", y_test.shape)


# ### 산점도

# In[28]:


# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 iris_dataset.feature_names에 있는 문자열을 사용합니다.
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()


# ### 첫 번째 머신러닝 모델: k-최근접 이웃 알고리즘

# In[29]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[30]:


knn.fit(X_train, y_train)


# ### 예측하기

# In[31]:


X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)


# In[32]:


prediction = knn.predict(X_new)
print("예측:", prediction)
print("예측한 타깃의 이름:",
     iris_dataset['target_names'][prediction])


# ### 모델 평가하기

# In[33]:


y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n", y_pred)


# In[34]:


print("테스트 세트의 정확도:{:.2f}".format(knn.score(X_test, y_test)))


# ### 요약 및 정리

# In[35]:


X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state = 0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))

