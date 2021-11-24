---
layout: post
title: "Customer Clsutering"
subtitle: " Clustering customers based income and amount spent on products"
date: 2020-11-24 10:12:13 -0400
background: '/img/posts/06.jpg'
my_variable: footer.html
---

Companies are investing and exploring strategeis designed maintain current customers, to acquire new customers, help retain the current base and increase the customers lifelong value. As competition is rising, customer relationship management plays a significant role in identifying and performing analysis of company's valuable customers and adopting best marketing strategies. This project is the illustration of using a clustering technique that identifies customers with similar characteristics and behaviors and segregating into homogeneous clusters. We assume that those distinct groups of customers who function differently and follow different approaches in their spending and purchasing habits. So main aim of the project is to identify different customer types and segment them into cluster of similar profiles, so target marketing can be executed effectively and efficiently. As a result, will develop high-quality and long-term customer relationship that increase loyalty, growth and profit. 



On this project, we will be using clustering algorithms KMeans Clustering. Clustering is a type of data mining technique used in a number of ways involving areas such as machine learning, pattern recognition and classification. 



Our dataset has information about Mall visitors such as income, total amount spent on certain products etc. Through KMeans algoriths, we will separate those customers into several clusters. Further marketing department can offer customized offers on products aimed at increasing sales. So our algorith builds clustering model of given dataset. 

Once the model have been fit to previously seen data they can be used to predict and understand new observations. 

We have data of 2249 customers visiting stores with following information


```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(suppress=True)
```


```python
df = pd.read_csv('marketing_campaign.csv', sep=';')
df.head()
```

<p> Below are features we have </p>

- Education level
- Marital status
- Kids at home
- Teen at home
- Income
- Amounts spent on fish products
- Amounts spent on mean products
- Amounts spent on fruits
- Amounts spent on sweet products 
- Amounts spent on gold products
- Amounts spent on wines
- Number of purchases made with discounts
- Number of purchases made with catalogue
- Number of purchases made in store
- Website purchases
- Number of visits to website
- Number of days since the last purchase

We also have data on customer acceptance of campaign 1 to 5. 

Now let’s see the shape of our data and information about the datafram including the data type of each column and memory usage of the entire data. 

    df.shape
    
<p> Data contains 2240 rows and 28 columns </p>


This info() pandas method prints information about dataframe incluyding the data types and non-null values

    df.info()


     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   Year_Birth           2240 non-null   int64  
     1   Education            2240 non-null   object 
     2   Marital_Status       2240 non-null   object 
     3   Income               2216 non-null   float64
     4   Kidhome              2240 non-null   int64  
     5   Teenhome             2240 non-null   int64  
     6   Dt_Customer          2240 non-null   object 
     7   Recency              2240 non-null   int64  
     8   MntWines             2240 non-null   int64  
     9   MntFruits            2240 non-null   int64  
     10  MntMeatProducts      2240 non-null   int64  
     11  MntFishProducts      2240 non-null   int64  
     12  MntSweetProducts     2240 non-null   int64  
     13  MntGoldProds         2240 non-null   int64  
     14  NumDealsPurchases    2240 non-null   int64  
     15  NumWebPurchases      2240 non-null   int64  
     16  NumCatalogPurchases  2240 non-null   int64  
     17  NumStorePurchases    2240 non-null   int64  
     18  NumWebVisitsMonth    2240 non-null   int64  
     19  AcceptedCmp3         2240 non-null   int64  
     20  AcceptedCmp4         2240 non-null   int64  
     21  AcceptedCmp5         2240 non-null   int64  
     22  AcceptedCmp1         2240 non-null   int64  
     23  AcceptedCmp2         2240 non-null   int64  
     24  Complain             2240 non-null   int64  
     25  Z_CostContact        2240 non-null   int64  
     26  Z_Revenue            2240 non-null   int64  
     27  Response             2240 non-null   int64  
    dtypes: float64(1), int64(24), object(3)
    memory usage: 490.1+ KB

Describe method gives us descriptive statistics of a dataframe. 

    df.describe().T



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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Year_Birth</th>
      <td>2240.0</td>
      <td>1968.805804</td>
      <td>11.984069</td>
      <td>1893.0</td>
      <td>1959.00</td>
      <td>1970.0</td>
      <td>1977.00</td>
      <td>1996.0</td>
    </tr>
    <tr>
      <th>Income</th>
      <td>2216.0</td>
      <td>52247.251354</td>
      <td>25173.076661</td>
      <td>1730.0</td>
      <td>35303.00</td>
      <td>51381.5</td>
      <td>68522.00</td>
      <td>666666.0</td>
    </tr>
    <tr>
      <th>Kidhome</th>
      <td>2240.0</td>
      <td>0.444196</td>
      <td>0.538398</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Teenhome</th>
      <td>2240.0</td>
      <td>0.506250</td>
      <td>0.544538</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Recency</th>
      <td>2240.0</td>
      <td>49.109375</td>
      <td>28.962453</td>
      <td>0.0</td>
      <td>24.00</td>
      <td>49.0</td>
      <td>74.00</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>MntWines</th>
      <td>2240.0</td>
      <td>303.935714</td>
      <td>336.597393</td>
      <td>0.0</td>
      <td>23.75</td>
      <td>173.5</td>
      <td>504.25</td>
      <td>1493.0</td>
    </tr>
    <tr>
      <th>MntFruits</th>
      <td>2240.0</td>
      <td>26.302232</td>
      <td>39.773434</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>8.0</td>
      <td>33.00</td>
      <td>199.0</td>
    </tr>
    <tr>
      <th>MntMeatProducts</th>
      <td>2240.0</td>
      <td>166.950000</td>
      <td>225.715373</td>
      <td>0.0</td>
      <td>16.00</td>
      <td>67.0</td>
      <td>232.00</td>
      <td>1725.0</td>
    </tr>
    <tr>
      <th>MntFishProducts</th>
      <td>2240.0</td>
      <td>37.525446</td>
      <td>54.628979</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>12.0</td>
      <td>50.00</td>
      <td>259.0</td>
    </tr>
    <tr>
      <th>MntSweetProducts</th>
      <td>2240.0</td>
      <td>27.062946</td>
      <td>41.280498</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>8.0</td>
      <td>33.00</td>
      <td>263.0</td>
    </tr>
    <tr>
      <th>MntGoldProds</th>
      <td>2240.0</td>
      <td>44.021875</td>
      <td>52.167439</td>
      <td>0.0</td>
      <td>9.00</td>
      <td>24.0</td>
      <td>56.00</td>
      <td>362.0</td>
    </tr>
    <tr>
      <th>NumDealsPurchases</th>
      <td>2240.0</td>
      <td>2.325000</td>
      <td>1.932238</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>3.00</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>NumWebPurchases</th>
      <td>2240.0</td>
      <td>4.084821</td>
      <td>2.778714</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>4.0</td>
      <td>6.00</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>NumCatalogPurchases</th>
      <td>2240.0</td>
      <td>2.662054</td>
      <td>2.923101</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.0</td>
      <td>4.00</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>NumStorePurchases</th>
      <td>2240.0</td>
      <td>5.790179</td>
      <td>3.250958</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>5.0</td>
      <td>8.00</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>NumWebVisitsMonth</th>
      <td>2240.0</td>
      <td>5.316518</td>
      <td>2.426645</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>AcceptedCmp3</th>
      <td>2240.0</td>
      <td>0.072768</td>
      <td>0.259813</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>AcceptedCmp4</th>
      <td>2240.0</td>
      <td>0.074554</td>
      <td>0.262728</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>AcceptedCmp5</th>
      <td>2240.0</td>
      <td>0.072768</td>
      <td>0.259813</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>AcceptedCmp1</th>
      <td>2240.0</td>
      <td>0.064286</td>
      <td>0.245316</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>AcceptedCmp2</th>
      <td>2240.0</td>
      <td>0.013393</td>
      <td>0.114976</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Complain</th>
      <td>2240.0</td>
      <td>0.009375</td>
      <td>0.096391</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Z_CostContact</th>
      <td>2240.0</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Z_Revenue</th>
      <td>2240.0</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>11.0</td>
      <td>11.00</td>
      <td>11.0</td>
      <td>11.00</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>Response</th>
      <td>2240.0</td>
      <td>0.149107</td>
      <td>0.356274</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Lets see if the dataset has any categorical data


```python
df.describe(include='O').T
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
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Education</th>
      <td>2240</td>
      <td>5</td>
      <td>Graduation</td>
      <td>1127</td>
    </tr>
    <tr>
      <th>Marital_Status</th>
      <td>2240</td>
      <td>8</td>
      <td>Married</td>
      <td>864</td>
    </tr>
    <tr>
      <th>Dt_Customer</th>
      <td>2240</td>
      <td>663</td>
      <td>2012-08-31</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




<p></p>

#### Feature Engineering 

Our features must not contain missing or null values and outliers. 

As part of feature engineering, we will check our data for missing or null values. We can find if our dataset contains any null values by running following line of code: 



```python
df.isnull().sum() 
```




    Year_Birth              0
    Education               0
    Marital_Status          0
    Income                 24
    Kidhome                 0
    Teenhome                0
    Dt_Customer             0
    Recency                 0
    MntWines                0
    MntFruits               0
    MntMeatProducts         0
    MntFishProducts         0
    MntSweetProducts        0
    MntGoldProds            0
    NumDealsPurchases       0
    NumWebPurchases         0
    NumCatalogPurchases     0
    NumStorePurchases       0
    NumWebVisitsMonth       0
    AcceptedCmp3            0
    AcceptedCmp4            0
    AcceptedCmp5            0
    AcceptedCmp1            0
    AcceptedCmp2            0
    Complain                0
    Z_CostContact           0
    Z_Revenue               0
    Response                0
    dtype: int64



We can see that income contains 24 missing values. We either can drop those missing values. However in our case, we will calculate the average of income and replace with average. This should not significantly affect our prediction. 


```python
df['Income']=df['Income'].fillna(df['Income'].mean())
df.isnull().sum()
```



    Year_Birth             0
    Education              0
    Marital_Status         0
    Income                 0
    Kidhome                0
    Teenhome               0
    Dt_Customer            0
    Recency                0
    MntWines               0
    MntFruits              0
    MntMeatProducts        0
    MntFishProducts        0
    MntSweetProducts       0
    MntGoldProds           0
    NumDealsPurchases      0
    NumWebPurchases        0
    NumCatalogPurchases    0
    NumStorePurchases      0
    NumWebVisitsMonth      0
    AcceptedCmp3           0
    AcceptedCmp4           0
    AcceptedCmp5           0
    AcceptedCmp1           0
    AcceptedCmp2           0
    Complain               0
    Z_CostContact          0
    Z_Revenue              0
    Response               0
    dtype: int64



Next step would be to check if there are any outliers in our dataset. We'll write detect_outlier function to detect our outliers. 


```python
def detect_outliers(frame):
    for i in frame.columns:
        if(frame[i].dtype == 'int64'):
            sns.boxplot(frame[i])
            plt.show()
            
        elif(frame[i].dtype == 'float64'):
            sns.boxplot(frame[i])
            plt.show()
            
detect_outliers(df)
```

![png](\posts\clustering\incomeout.png)


We can see that income has extremely high value that is outlier. Such outlier has to be removed as we are not expecting small number of visitor with extremely high income. 


We have several options in dealing with outliers. In our case, we will replace missing values with mean value of salary. This is reasoable approach and should not bias our outcome significantly. 


```python
df=df[np.abs(df.Income-df.Income.mean())<=(3*df.Income.std())]
sns.boxplot(data=df['Income'])
```


    
![png](\posts\clustering\incomewithoutout.png)
    


Now outliers have been removed. Please refer to above output


Lets take a look at correlation matrix between features.

    plt.figure(figsize=(30,30))
    sns.heatmap(df.corr(),annot=True)

![png](\posts\clustering\correl.png)

#### Modeling
KMeans clustering is one of the most poplular algorithms used for clustering as its simple and efficient to use. The aim of the KMeans algorithm is to divide M points in N dimensions into K clusters fixed a priori. K cluster points that will be centroids are placed in the space among the data points and each data point is assigned to the centroid for which the distance is the least. This means that algorithm will be completed when objective function will have least squarred error. 

KMeans clustering requires number of clusters that we need to input. In order to identify number of clusters, we will use elbow methods that will help us to get the optimal number of clusters recommended. 

    X1=df[['Income','TotalSpent']].iloc[:,:].values
    clusters=[]
    for i in range(1,11): 
            kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
            kmeans.fit(X1)
            clusters.append(kmeans.inertia_)
    plt.plot(range(1,11),clusters)
    plt.title('Elbow Method')
    plt.xlabel('no of clusters')
    plt.ylabel('Inertia')
    plt.show() 

In figure below, it can be observed that the elbow point occurs at K=4. After K=4, the differences is not significant. Hence we selected K=4 clusters and will use it as an input in our KMeans algorithm. 

![png](\posts\clustering\elbow.png)



    X=df[['Income','TotalSpent']]

    km_5 = KMeans(n_clusters=4, init='k-means++', random_state=0)
    km_5.fit(X)
    centroids = km_5.cluster_centers_
    X['Labels'] = km_5.labels_

    plt.figure(figsize=(12, 8))

    sns.scatterplot(X['Income'], X['TotalSpent'], hue=X['Labels'], 
                    palette=sns.color_palette('hls', 5))

    plt.scatter(centroids[:,0], centroids[:,1], c='red',s=200)

    plt.title('',fontsize=18)
    plt.show()


![png](\posts\clustering\clustimg.png)

As shown above, the scatter plot of the clusters is created with Income on Y-axis against TotalSpent on X-axis. The datapoints under each cluster are represented using different colors and the centroids are also depicted. 

Based on this clustering, each group's values have to be studied and marketing department has to develop proper strategies aimed at increasing customer loyalty and profits. 
