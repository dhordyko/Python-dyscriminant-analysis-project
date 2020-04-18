```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import os
import warnings
```


```python

#Załadowanie zbioru danych i udzelienie nazw atrybutom
dataset=pd.read_table('heart.dat',sep="\s+",names =['age', 
                                                    'sex' ,
                                                    'cp',
                                                    'trestbps',
                                                    'chol',
                                                    'fbs',
                                                    'restecg',
                                                    'thalach',
                                                    'exang',
                                                    'oldpeak',
                                                    'slope',
                                                    'ca',
                                                    'thal',
                                                    'target'])

```


```python
#Wyświetlenie pierwszych wierszy zbioru
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>70.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>130.0</td>
      <td>322.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>109.0</td>
      <td>0.0</td>
      <td>2.4</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>115.0</td>
      <td>564.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>160.0</td>
      <td>0.0</td>
      <td>1.6</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>57.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>124.0</td>
      <td>261.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>141.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>128.0</td>
      <td>263.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>105.0</td>
      <td>1.0</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>120.0</td>
      <td>269.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>121.0</td>
      <td>1.0</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Wyświetlenie ostatnich wierszy zbioru
dataset.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>265</th>
      <td>52.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>172.0</td>
      <td>199.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>266</th>
      <td>44.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>120.0</td>
      <td>263.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>173.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>267</th>
      <td>56.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>140.0</td>
      <td>294.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>153.0</td>
      <td>0.0</td>
      <td>1.3</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>268</th>
      <td>57.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>140.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>148.0</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>269</th>
      <td>67.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>160.0</td>
      <td>286.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>108.0</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Ogólna informacja o zbiorze 
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 270 entries, 0 to 269
    Data columns (total 14 columns):
    age         270 non-null float64
    sex         270 non-null float64
    cp          270 non-null float64
    trestbps    270 non-null float64
    chol        270 non-null float64
    fbs         270 non-null float64
    restecg     270 non-null float64
    thalach     270 non-null float64
    exang       270 non-null float64
    oldpeak     270 non-null float64
    slope       270 non-null float64
    ca          270 non-null float64
    thal        270 non-null float64
    target      270 non-null int64
    dtypes: float64(13), int64(1)
    memory usage: 29.7 KB
    


```python
#Wykres korelacji między atrybutami zbioru 
plt.figure(figsize=(10,10))
sns.heatmap(dataset.corr(),vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
plt.tight_layout()
plt.show()
```


![png](output_5_0.png)



```python
#Wykres rozproszenia danych według atrybutów Wiek i Płeć 
dataset['AgeRange']=0
youngAge_index=dataset[(dataset.age>=29)&(dataset.age<40)].index
middleAge_index=dataset[(dataset.age>=40)&(dataset.age<55)].index
elderlyAge_index=dataset[(dataset.age>55)].index
for index in elderlyAge_index:
    dataset.loc[index,'AgeRange']=2
    
for index in middleAge_index:
    dataset.loc[index,'AgeRange']=1

for index in youngAge_index:
    dataset.loc[index,'AgeRange']=0
sns.swarmplot(x="AgeRange", y="age",hue='sex',
              palette=["r", "c", "y"], data=dataset)
plt.show()
```


![png](output_6_0.png)



```python
#Wykres rozproszenia danych według atrybutu docelowego 
sns.countplot(dataset.target,hue=dataset.target)
plt.title("Posiadanie albo brak choroby serca")
plt.show()
```


![png](output_7_0.png)



```python
#Wykres grup wiekowych
colors = ['pink','blue','yellow']
explode = [0,0,0.1]
plt.figure(figsize = (5,5))

plt.pie([len(young_ages),len(middle_ages),len(elderly_ages)],labels=['young ages','middle ages','elderly ages'],explode=explode,colors=colors, autopct='%1.1f%%')
plt.title('Wiek',color = 'blue',fontsize = 18)
plt.show()
```


![png](output_8_0.png)



```python
dataset[dataset['target']==1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
      <th>AgeRange</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>67.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>115.0</td>
      <td>564.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>160.0</td>
      <td>0.0</td>
      <td>1.6</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>128.0</td>
      <td>263.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>105.0</td>
      <td>1.0</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>120.0</td>
      <td>269.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>121.0</td>
      <td>1.0</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>65.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>120.0</td>
      <td>177.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>140.0</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>59.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>135.0</td>
      <td>234.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>161.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>263</th>
      <td>49.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>130.0</td>
      <td>266.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>171.0</td>
      <td>0.0</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>265</th>
      <td>52.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>172.0</td>
      <td>199.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>266</th>
      <td>44.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>120.0</td>
      <td>263.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>173.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>267</th>
      <td>56.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>140.0</td>
      <td>294.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>153.0</td>
      <td>0.0</td>
      <td>1.3</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>268</th>
      <td>57.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>140.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>148.0</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 15 columns</p>
</div>




```python
plt.figure(figsize=(10,10))
sns.heatmap(dataset[dataset['target']==2].corr(),vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
plt.tight_layout()
plt.show()
```


![png](output_10_0.png)



```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import style
import pandas as pd 
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

```


```python
dataset=pd.read_table('heart.dat',sep="\s+",names =['age', 
                                                    'sex' ,
                                                    'cp',
                                                    'trestbps',
                                                    'chol',
                                                    'fbs',
                                                    'restecg',
                                                    'thalach',
                                                    'exang',
                                                    'oldpeak',
                                                    'slope',
                                                    'ca',
                                                    'thal',
                                                    'target'])

```


```python
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>70.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>130.0</td>
      <td>322.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>109.0</td>
      <td>0.0</td>
      <td>2.4</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>115.0</td>
      <td>564.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>160.0</td>
      <td>0.0</td>
      <td>1.6</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>57.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>124.0</td>
      <td>261.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>141.0</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>128.0</td>
      <td>263.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>105.0</td>
      <td>1.0</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>120.0</td>
      <td>269.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>121.0</td>
      <td>1.0</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>265</th>
      <td>52.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>172.0</td>
      <td>199.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>266</th>
      <td>44.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>120.0</td>
      <td>263.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>173.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>267</th>
      <td>56.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>140.0</td>
      <td>294.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>153.0</td>
      <td>0.0</td>
      <td>1.3</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>268</th>
      <td>57.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>140.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>148.0</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>269</th>
      <td>67.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>160.0</td>
      <td>286.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>108.0</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>270 rows × 14 columns</p>
</div>




```python
dataset.info
```




    <bound method DataFrame.info of       age  sex   cp  trestbps   chol  fbs  restecg  thalach  exang  oldpeak  \
    0    70.0  1.0  4.0     130.0  322.0  0.0      2.0    109.0    0.0      2.4   
    1    67.0  0.0  3.0     115.0  564.0  0.0      2.0    160.0    0.0      1.6   
    2    57.0  1.0  2.0     124.0  261.0  0.0      0.0    141.0    0.0      0.3   
    3    64.0  1.0  4.0     128.0  263.0  0.0      0.0    105.0    1.0      0.2   
    4    74.0  0.0  2.0     120.0  269.0  0.0      2.0    121.0    1.0      0.2   
    ..    ...  ...  ...       ...    ...  ...      ...      ...    ...      ...   
    265  52.0  1.0  3.0     172.0  199.0  1.0      0.0    162.0    0.0      0.5   
    266  44.0  1.0  2.0     120.0  263.0  0.0      0.0    173.0    0.0      0.0   
    267  56.0  0.0  2.0     140.0  294.0  0.0      2.0    153.0    0.0      1.3   
    268  57.0  1.0  4.0     140.0  192.0  0.0      0.0    148.0    0.0      0.4   
    269  67.0  1.0  4.0     160.0  286.0  0.0      2.0    108.0    1.0      1.5   
    
         slope   ca  thal  target  
    0      2.0  3.0   3.0       2  
    1      2.0  0.0   7.0       1  
    2      1.0  0.0   7.0       2  
    3      2.0  1.0   7.0       1  
    4      1.0  1.0   3.0       1  
    ..     ...  ...   ...     ...  
    265    1.0  0.0   7.0       1  
    266    1.0  0.0   7.0       1  
    267    2.0  0.0   3.0       1  
    268    2.0  0.0   6.0       1  
    269    2.0  3.0   3.0       2  
    
    [270 rows x 14 columns]>




```python
df_by_class = dataset.groupby('target').count()
```


```python
df_by_class
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
    </tr>
    <tr>
      <th>target</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>150</td>
      <td>150</td>
      <td>150</td>
      <td>150</td>
      <td>150</td>
      <td>150</td>
      <td>150</td>
      <td>150</td>
      <td>150</td>
      <td>150</td>
      <td>150</td>
      <td>150</td>
      <td>150</td>
    </tr>
    <tr>
      <th>2</th>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Podział zbioru na uczący i testujący 
X = dataset.iloc[:, 0:12].values
y=dataset.iloc[:,13].values
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
#Noramlizacja zbioru
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Zastosowanie funkcji QDA dla budowy modeli
qda = QuadraticDiscriminantAnalysis()


   
    
#Trenowanie modeli na zbiorze uczącym 
model = qda.fit(X_train, y_train)
#Zastosowanie modeli do klasyfikacji zbioru testowego
pred=model.predict(X_test)
from sklearn import metrics
#Wyliczenie błendu klasyfikacji
print("Accuracy")
print(metrics.accuracy_score(y_test, pred))

print(np.unique(pred, return_counts=True))
print(confusion_matrix(pred, y_test))
print(classification_report(y_test, pred, digits=3))
```


```python
X_test.shape
```




    (68, 12)




```python
def BoundaryLine(kernel, algo, algo_name):
    reduction = KernelPCA(n_components=2, kernel = kernel)
    x_train_reduced = reduction.fit_transform(X_train)
   
    x_test_reduced = reduction.transform(X_test)
    
    classifier = algo
    classifier.fit(x_train_reduced, y_train)
    
    y_pred = classifier.predict(x_test_reduced)
    X_set, y_set = np.concatenate([x_train_reduced, x_test_reduced], axis = 0), np.concatenate([y_train, y_pred], axis = 0)
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.5, cmap = matplotlib.colors.ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = matplotlib.colors.ListedColormap(('red', 'green'))(i), label = j)
   
    plt.xticks(fontsize = 3)
    plt.yticks(fontsize = 3)
fig=plt.figure(figsize=(180, 160), dpi=80)
fig.suptitle('QDA Classifier')

ax = plt.subplot(7,5,1)
ax.set_title('QDA')
ax.set_ylabel('', rotation = 0, labelpad=30, fontsize = 0)
BoundaryLine('rbf',model, "QDA Model")
fig.show()
    
```

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    C:\Users\Darya\Anaconda3\lib\site-packages\ipykernel_launcher.py:31: UserWarning: Matplotlib is currently using module://ipykernel.pylab.backend_inline, which is a non-GUI backend, so cannot show the figure.
    


![png](output_19_1.png)



```python

```
