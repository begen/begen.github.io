---
layout: post
title: "Customer Segmentation"
subtitle: "Predicting customer segments that will respond to an offer for a product based on dataset"
date: 2021-09-10 10:12:13 -0400
background: '/img/posts/06.jpg'
my_variable: footer.html
---

<p> 
When it comes to the definition of ML, I want to thin as building models of data.</p>

<p>In essence, machine learning basically mean building mathematical models to help understand data. </p>

<p>Word “learning” is associated with tuning parameters. </p>

<p>Once the model have been fit to previously seen data they can be used to predict and understand new observations. </p>

<p>We have data of 2249 customers visiting stores with following information</p>

<p>
Education level
Marital status
Kids at home
Teen at home
Income
Amounts spent on fish products
Amounts spent on mean products
Amounts spent on fruits
Amounts spent on sweet products 
Amounts spent on gold products
Amounts spent on wines
Number of purchases made with discounts
Number of purchases made with catalogue
Number of purchases made in store
Website purchases
Number of visits to website
Number of days since the last purchase
</p>

<p>We also have data on customer acceptance of campaign 1 to 5. </p>
<p>Our target variable is response that is if customer accepted the offer in one of those campaigns. </p>

<p>In this research project, we will be identifying groups of people who will respond to an offer for a product or a service. </p> 

<p>Based on the data, we know that our target variable or dependent variable is “Response” column with 1 and 0 values. </p>

<p>Our predictive variables are the rest of columns that we explained above. Predictive variables are also called independent variables or features.</p>  



Lets first import data: 

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

    pd.options.display.max_columns = None

    pd.options.display.max_rows = None

    np.set_printoptions(suppress=True)


    df = pd.read_csv('marketing_campaign.csv', sep=';')
    df.head()

<p>We have some unnecessary column that is ID, we do not need this ID column, hence we drop. It from our dataframe </p>
    
    df.drop('ID', axis=1, inplace = True)
    print('Data contains', df.shape[0], 'rows and', df.shape[1], 'columns')
    df.info()
<p>To get descriptive statistics of the dataframe, we can run datagframe.describe() </p>
  
    df.describe().T
    df.describe(include='O').T
  
<p>Feature Engineering 

Our features (independent variables) must not contain missing or null values, outliers, data on a different scale, human errors and multicollinearity and very large independent sets. Multicollinearity is the concept when features are correlated. We do not want many features that are correlated to each other. 

As part of feature engineering, we will check our data for missing or null values. We can find if our dataset contains any null values by running following line of code: 
</p>
    df.isnull().sum() 
    df['Income']=df['Income'].fillna(df['Income'].mean())
    df.isnull().sum()

<p>We can see that income contains 24 missing values. We either can drop those missing values. However in our case, we will calculate the average of income and replace with average. This should not significantly affect our prediction.</p> 

    df=df[np.abs(df.Income-df.Income.mean())<=(3*df.Income.std())]
    sns.boxplot(data=df['Income'])


    sns.countplot(df['Response'])
    plt.show()

    #sns.pairplot(df)

    plt.figure(figsize=(30,25))
    sns.heatmap(df.corr(), annot=True)

    df['Dt_Customer'] = df['Dt_Customer'].astype('datetime64')
    df['Date_Customer'] = df['Dt_Customer'].dt.day
    df['Month_Customer'] = df['Dt_Customer'].dt.month
    df['Year_Customer'] = df['Dt_Customer'].dt.year
    df.drop('Dt_Customer', axis=1, inplace=True)
    df

    df_cat=df.select_dtypes(exclude=[np.number])
    df_cat

    def encode(dataframe):
        lec = LabelEncoder()
        for j in dataframe.columns:
            if(dataframe[j].dtype == 'object'):
                dataframe[j] = lec.fit_transform(dataframe[j])
                
    encode(df)
    df

    x = df.drop('Response', axis=1)
    y = df['Response']

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    log_reg = LogisticRegression(max_iter=5000)
    log_reg.fit(X_train, Y_train)

    log_reg_pred = log_reg.predict(X_test)
    print(classification_report(Y_test, log_reg_pred))








```python
df = pd.read_csv('marketing_campaign.csv', sep=';')

# Dropping ID Column beacause we dont id column for predictions
df.drop('ID', axis=1, inplace = True)

# Shape of Dataset
print('Data contains', df.shape[0], 'rows and', df.shape[1], 'columns')

# Dataset information about value count and variable data type
df.info()

df.describe().T

# Categorical Data Description
df.describe(include='O').T

# Check for null values in the dataset
df.isnull().sum() 


# Filling NA
def fill_na(frame):
    for i in frame.columns:
        if(((frame[i].isnull().sum() / len(frame))*100) <= 30) & (frame[i].dtype == 'int64'):
            frame[i] = frame[i].fillna(frame[i].median())
            
        elif(((frame[i].isnull().sum() / len(frame))*100) <= 30) & (frame[i].dtype == 'O'):
            frame[i] = frame[i].fillna(frame[i].mode()[0])
            
        elif(((frame[i].isnull().sum() / len(frame))*100) <= 30) & (frame[i].dtype == 'float64'):
            frame[i] = frame[i].fillna(frame[i].median())
            
fill_na(df)

#Checking outliers
def detect_outliers(frame):
    for i in frame.columns:
        if(frame[i].dtype == 'int64'):
            sns.boxplot(frame[i])
            plt.show()
            
        elif(frame[i].dtype == 'float64'):
            sns.boxplot(frame[i])
            plt.show()
            
detect_outliers(df)


# Plot Response variable seperately because our target variable(Class) is int and we have to treat it like object this time
sns.countplot(df['Response'])
plt.show()

sns.pairplot(df)

# Check correlation between variables
plt.figure(figsize=(30,25))
sns.heatmap(df.corr(), annot=True)

df['Dt_Customer'] = df['Dt_Customer'].astype('datetime64')
# Creating two new columns Date_customer and Month_customer from Dt_Customer column
df['Date_Customer'] = df['Dt_Customer'].dt.day
df['Month_Customer'] = df['Dt_Customer'].dt.month
df['Year_Customer'] = df['Dt_Customer'].dt.year
# Now we can drop Dt_Customer column
df.drop('Dt_Customer', axis=1, inplace=True)

def encode(dataframe):
    lec = LabelEncoder()
    for j in dataframe.columns:
        if(dataframe[j].dtype == 'object'):
            dataframe[j] = lec.fit_transform(dataframe[j])
            
encode(df)

x = df.drop('Response', axis=1)
y = df['Response']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1)
```


```python
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, Y_train)

lr_pred = lr.predict(X_test)
print(classification_report(Y_test, lr_pred))

```


```python
xgb = XGBClassifier()
xgb.fit(X_train, Y_train)

xgb_pred=xgb.predict(X_test)
print(classification_report(Y_test,xgb_pred))

```


```python
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

rf_pred = rf.predict(X_test)
print(classification_report(Y_test, rf_pred))
```


```python
gb=GradientBoostingClassifier()
gb.fit(X_train,Y_train)

gb_pred=gb.predict(X_test)
print(classification_report(Y_test,gb_pred))

accuracy_score(Y_test,gb_pred)

```


```python
estimators=[('xgb',XGBClassifier()),
('rf', RandomForestClassifier()),
('gb',GradientBoostingClassifier())]
stack=StackingClassifier(estimators=estimators)
stack.fit(X_train,Y_train)

stack_pred=stack.predict(X_test)
print(classification_report(Y_test,stack_pred))
accuracy_score(Y_test,stack_pred)

```


```python
# Balancing the target variable
smote=SMOTETomek()
x_train,y_train=smote.fit_resample(X_train,Y_train)

```


```python
#Logistics regression
slr=LogisticRegression(max_iter=10000)
slr.fit(x_train,y_train)
```


```python
slr_pred=slr.predict(X_test)
print(classification_report(Y_test,slr_pred))
accuracy_score(Y_test,slr_pred)
```


```python
#XGBoost Classifier

sxgb=XGBClassifier()
sxgb.fit(x_train,y_train)

sxgb_pred=sxgb.predict(X_test)
print(classification_report(Y_test, sxgb_pred))
accuracy_score(Y_test,slr_pred)
```


```python
srf=RandomForestClassifier()
srf.fit(x_train,y_train)

sfr_pred=srf.predict(X_test)
print(classification_report(Y_test,sfr_pred))
accuracy_score()
```


```python
sgb=GradientBoostingClassifier()
sgb.fit(x_train,y_train)

sgb_pred=sgb.predict(X_test)
print(classification_report(Y_test,sgb_pred))
accuracy_score(Y_test,sgb_pred)
```


```python
sstack = StackingClassifier(estimators=estimators)
sstack.fit(x_train, y_train)

sstack_pred = sstack.predict(X_test)
print(classification_report(Y_test, sstack_pred))

```

### Future Selection


```python
th = np.sort(gb.feature_importances_)
l = []
for g in th:
    select = SelectFromModel(gb, threshold = g, prefit = True)
    x_Train = select.transform(X_train)
    model = GradientBoostingClassifier()
    model.fit(x_Train, Y_train)
    x_Test = select.transform(X_test)
    y_pred = model.predict(x_Test)
    accuracy = accuracy_score(Y_test, y_pred)
    print('Threshold:', g, 'Model Score:', accuracy)
```


```python
imp = pd.DataFrame(rf.feature_importances_)
imp.index = X_train.columns
imp[imp[0] < 0.017037885998921535]
```


```python
X_train = X_train.drop(['Z_CostContact', 'Z_Revenue'], axis=1)
X_test = X_test.drop(['Z_CostContact', 'Z_Revenue'], axis=1)
```


```python
fgb = GradientBoostingClassifier()
fgb.fit(X_train, Y_train)
```


```python
fgb_pred = fgb.predict(X_test)
print(classification_report(Y_test, fgb_pred))
```


```python
accuracy_score(Y_test, fgb_pred)
```


```python
# First i check how many components we want
# For this first i am initializing the pca
pca = PCA()
# Fitting the training set in pca
pca.fit(X_train)
```


```python
# Now check number of components
pca.explained_variance_ratio_
```


```python
# Creating pca with n_components = 15
Pca = PCA(n_components=15)
# Fitting the training data
X_Train = Pca.fit_transform(X_train)
X_Test = Pca.fit_transform(X_test)
```


```python
# Building models after applying pca
pgb = GradientBoostingClassifier()
pgb.fit(X_Train, Y_train)
```

```python
pgb_pred = pgb.predict(X_Test)
print(classification_report(Y_test, pgb_pred))
```

```python
grid = {
    'learning_rate' : [0.2, 0.3, 0.4, 0.5],
    'n_estimators' : [300, 500, 700, 900],
    'min_samples_split' : [3, 4, 5, 6],
    'max_depth' : [2, 3, 4, 5],
    'loss' : ['deviance', 'exponential']
}
random_cv = RandomizedSearchCV(estimator=gb,
                              param_distributions=grid,
                              n_iter=20,
                              n_jobs=-1,
                              cv=5,
                              verbose=7,
                              random_state=10,
                              scoring='accuracy')
random_cv.fit(X_train, Y_train)
```


```python
random_cv.best_estimator_
```


```python
hgb = GradientBoostingClassifier(learning_rate=0.5, loss='exponential', max_depth=2,
                           min_samples_split=4, n_estimators=300)
hgb.fit(X_train, Y_train)
```


```python
hgb_pred = hgb.predict(X_test)
print(classification_report(Y_test, hgb_pred))
accuracy_score(Y_test, hgb_pred)
```


```python

```

My Best model is Gradient Boosting Classifier after Hyper Parameter tuning




