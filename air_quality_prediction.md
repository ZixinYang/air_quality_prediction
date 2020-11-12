***Author: 楊子欣(Tzu-Hsin Yang)***
[CV](https://drive.google.com/file/d/1JPVD1KnS9iokhlYCguCRLz4MgqOKe-hm/view?usp=sharing)      [Github](https://github.com/ZixinYang)     [Linkedin](https://www.linkedin.com/in/tzuhsinyang/)



# Air Quality Prediction
Implemented on [jupyter notebook](https://github.com/ZixinYang/air_quality_prediction/blob/master/air_quality_prediction.ipynb) 
(sometimes github fails to render the ipynb notebooks, so move it to HackMD)

Dataset: https://archive.ics.uci.edu/ml/datasets/Air+Quality
***Missing values are tagged with -200 value.***

## Data Preprocessing
* Drop Datetime feature column (the order already contain time series information)
* Drop all-nan rows and columns
* Convert string type into float type
* Calculate how many -200(nan) exists in each column, and drop the column which contains too many missing data
* Missing Data Imputation: (1) Plot correlation matrix(after filter -200 value) (2) Use Linear Regression to imputate missing data with high correlation feature (3) If regression score is smaller than 0.8, use mean instead of regression result
* Since EPA Administrator uses CO, NO2, PM2.5, PM10, O3, SO2 as standard of Air Quality Index and use maximum of their conversion as the AQI value, CO and NO2 are selected as air quality definition which is max(AQI(CO),AQI(NO2))
* EPA Administrator evaluates the concentration of CO on 8 hourly average, so the values are assigned with 8 hourly average value
* Plot quartiles graph to see if there are many outliers in CO and NO2 data. CO and NO2 has 301 and 380 outliers, respectively. Since it only account for a few ratio, it is no need for removing. Also, when we see the exact value, they are in common range of AQI.
* The time series curve is also be plotted. The pattern of CO is more clear than the plot of NO2. NO2's curve seems to be more dramatical.
* Convert the concentration of CO and NO2 into their AQI. However, we can see that only few CO's index is larger than NO2's index. Therefore, after AQI column is built, NO2 column is removed because we don't need two highly similar columns to train the model.

### Import Library
```python=
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
%matplotlib inline
```
```python=
# read data
def readTrain(filename, skipped):
    train = pd.read_csv(filename, sep=';',usecols=lambda x: x not in skipped)
    return train
 
# drop Date and Time since the order already contain time series information
columns_to_skip = ['Date','Time']

df = readTrain('air_data.csv', columns_to_skip)
df = df.dropna(how='all')
df = df.dropna(axis='columns')

# observe the type of each column
for c in df.columns:
    print(type(df[c][0]))

# some of column contains float but use comma, convert it to float
for c in df.columns:
    if c!='Date' and c!='Time' and isinstance(df[c][0], str):
        df[c] = df[c].apply(lambda x: float(x.replace(',','.')))

# count how many -200 value in each column
idx = []
nan_row = {'CO(GT)':0,'PT08.S1(CO)':0,'NMHC(GT)':0,'C6H6(GT)':0,'PT08.S2(NMHC)':0,'NOx(GT)':0,'PT08.S3(NOx)':0,'NO2(GT)':0,'PT08.S4(NO2)':0,'PT08.S5(O3)':0,'T':0,'RH':0,'AH':0}
for index, row in df.iterrows():
    for c in nan_row.keys():
        if row[c]==-200:
            nan_row[c]+=1
            idx.append(index)
print(nan_row)

# NMHC(GT) contains 8443 -200 value, drop it beacause it is not possible to do the imputaion for this column
df = df.drop(columns=['NMHC(GT)'])
```
```python=
# find out the rows which don't contain any -200
df_ = df.drop(list(set(idx)))
alpha = list(df_.columns)

# plot the correlation matrix between features
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df_.corr(), interpolation='nearest')
fig.colorbar(cax)

ax.set_xticklabels(['']+alpha)
ax.set_yticklabels(['']+alpha)
plt.show()
```
![](https://i.imgur.com/8bwQvXt.png)

```python=
# print out the correlation between each features and select highest correlation feature for each
tmp = df_.corr().abs()
for c in df_.columns:
    print(tmp[c].sort_values())
```
```python=
# pair highest correlation for each feature
pair=[['CO(GT)','C6H6(GT)'],\
     ['PT08.S1(CO)','PT08.S4(NO2)'],\
     ['C6H6(GT)','PT08.S2(NMHC)'],\
     ['PT08.S2(NMHC)','C6H6(GT)'],\
     ['NOx(GT)','CO(GT)'],\
     ['PT08.S3(NOx)','PT08.S2(NMHC)'],\
     ['NO2(GT)','PT08.S2(NMHC)'],\
     ['PT08.S4(NO2)','C6H6(GT)'],\
     ['PT08.S5(O3)','PT08.S1(CO)']]
```
```python=
# use linear regression to predict the missing data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
miss = dict.fromkeys(df.columns)
score = dict.fromkeys(df.columns)
for p in pair:
    print(p)
    df_tmp = pd.DataFrame(df, columns=p)
    df_tmp = df_tmp.loc[df_tmp[p[0]]!=-200]
    df_tmp = df_tmp.loc[df_tmp[p[1]]!=-200]
    
    X = np.array(list(df_tmp[p[1]])).reshape(-1, 1)
    y = np.array(list(df_tmp[p[0]])).reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    score[p[0]] = reg.score(X,y)
    print(reg.score(X,y))
    
    df_tmp = pd.DataFrame(df, columns=p)
    df_tmp = df_tmp.loc[df_tmp[p[0]] ==-200]
    miss[p[0]] = reg.predict(np.array(list(df_tmp[p[1]])).reshape(-1, 1))
```
```python=
# if reg score is higher than 0.8, assign the values
for c in df.columns:
    if score[c] and score[c]>=0.8:
        x = 0
        for i in range(len(df)):
            if df[c][i]==-200:
                df.loc[i,p[0]] = miss[c][x]
                x+=1
```
```python=
# if reg score is smaller than 0.8, assign the mean of the feature
for c in df.columns:
    count = 0
    for x in list(df[c]):
        if x==-200:
            count+=1
    fake_mean = np.mean(df[c])
    true_mean = (fake_mean*len(df)-(-200)*count)/(len(df)-nan_row[c])
    df[c] = df[c].apply(lambda x: true_mean if x == -200 else x)
```

### Take CO and NO2 as targets
```python=
# reassign CO's value with 8 hourly average value (follow the standard of EPA Administrator)
for i in range(len(df)):
    if i<8:
        df.loc[i,'CO(GT)'] = np.mean(df['CO(GT)'][:i+1])
    else:
        df.loc[i,'CO(GT)'] = np.mean(df['CO(GT)'][i-8:i+1])
```
```python=
# plot the quartile box to observe the distribution of CO
import seaborn as sns
sns.boxplot(x=df['CO(GT)'])
```
![](https://i.imgur.com/yPeihCQ.png)
```python=
co = sorted(list(df['CO(GT)']))
Q1 = (co[int(0.25*len(co))]+co[int(0.25*len(co))+1])/2
Q3 = (co[int(0.75*len(co))]+co[int(0.75*len(co))+1])/2
outlier_top = Q3+(Q3-Q1)*1.5
outlier_buttom = Q1-(Q3-Q1)*1.5
print(outlier_buttom, outlier_top)
print(sum(1 for i in co if i > outlier_top)+sum(1 for i in co if i < outlier_buttom))
```
output:
0.9242777055787319 3.323680323992303
301
It shows that the outliers only account for 3% of data
```python=
# plot the quartile box to observe the distribution of NO2
sns.boxplot(x=df['NO2(GT)'])
```
![](https://i.imgur.com/MUen9Ol.png)
```python=
no2 = sorted(list(df['NO2(GT)']))
Q1 = (no2[int(0.25*len(no2))]+no2[int(0.25*len(no2))+1])/2
Q3 = (no2[int(0.75*len(no2))]+no2[int(0.75*len(no2))+1])/2
outlier_buttom = Q1-(Q3-Q1)*1.5
outlier_top = Q3+(Q3-Q1)*1.5
print(outlier_buttom, outlier_top)
print(sum(1 for i in no2 if i > outlier_top)+sum(1 for i in no2 if i < outlier_buttom))
```
output:
15.5 203.5
380
It shows that the outliers only account for 4% of data
```python=
# plot the time series of CO
plt.plot(range(len(df)),df['CO(GT)'])
```
![](https://i.imgur.com/R40IqKb.png)
```python=
# plot the time series of NO2
plt.plot(range(len(df)),normalize(np.array(df['NO2(GT)']).reshape(-1,1),axis=0).ravel())
```
![](https://i.imgur.com/eCnuw4s.png)
It shows that the NO2 varies more dramatically than CO

### The standard AQI of EPA Administrator 
|AQI|CO|NO2|
|---|---|---|
|0～50|0 - 4.4|0 - 53|
|51～100|4.5 - 9.4|54 - 100|
|101～150|9.5 - 12.4|101 - 360|
|151～200|12.5 - 15.4|361 - 649|
|201～300|15.5 - 30.4|650 - 1249|
|301～400|30.5 - 40.4|1250 - 1649|
|401～500|40.5 - 50.4|1650 - 2049|

### Convert the value to AQI (generate the label for training)
```python=
def return_interval(s, f):
    for i in range(1,len(MAP[s])):
        if f<MAP[s][i]:
            return i-1

MAP = {'AQI':[0,51,101,151,201,301,401,501],
       'CO':[0,4.5,9.5,12.5,15.5,30.5,40.5,50.5],
       'NO2':[0,54,101,361,650,1250,1650,2050]
    }
slope_intercept = {'CO':[],'NO2':[]}
for i in range(1,len(MAP['AQI'])):
    cx1 = MAP['CO'][i-1]
    nx1 = MAP['NO2'][i-1]
    y1 = MAP['AQI'][i-1]
    cx2 = MAP['CO'][i]-0.1
    nx2 = MAP['NO2'][i]-1
    y2 = MAP['AQI'][i]-1
    ca = (y2-y1)/(cx2-cx1)
    cb = y1-ca*cx1
    na = (y2-y1)/(nx2-nx1)
    nb = y1-na*nx1
    slope_intercept['CO'].append((ca,cb))
    slope_intercept['NO2'].append((na,nb))

co = list(df['CO(GT)'])
no2 = list(df['NO2(GT)'])
aqi = []
caqi = []
naqi = []
count = 0
for i in range(len(co)):
    ca, cb = slope_intercept['CO'][return_interval('CO',co[i])]
    na, nb = slope_intercept['NO2'][return_interval('NO2',no2[i])]
    c = ca*co[i]+cb
    n = na*no2[i]+nb
    caqi.append(c)
    naqi.append(n)
    MAX = max(c,n)
    aqi.append(MAX)
print(sum(1 for i in range(len(aqi)) if aqi[i]!=naqi[i]))
```
output:
45
It shows that mostly NO2 AQI is bigger than CO AQI.
That is, The AQI value is highly related to NO2 value.
```python=
df.insert(len(df.columns), "AQI", aqi, True)
# drop NO2 to avoid containing two highly related features
df = df.drop(columns=['NO2(GT)'])
df.to_csv('data.csv', sep=';')
```
![](https://i.imgur.com/9u0P0Uv.png)

## Training model
* Given previous 5-day data(input), we would like to predict AQI of the next 5 days(output)
* Use RNN(with LSTM) model to learn the time series pattern and predict the AQI

```python=
# split the training and testing set
def buildTrain(train, pastHour=120, futureHour=120):
    XX, yy = [], []
    from sklearn.preprocessing import normalize
    from sklearn.utils import shuffle
    for i in range(train.shape[0]-futureHour-pastHour):
        x = np.array(train.iloc[i:i+pastHour])
        x = normalize(x, axis=1)
        XX.append(x)
        yy.append(np.array(train.iloc[i+pastHour:i+pastHour+futureHour]["AQI"]))
    XX, yy = shuffle(XX, yy)
    return np.array(XX[:int(len(XX)*0.7)]), np.array(yy[:int(len(yy)*0.7)]), np.array(XX[int(len(XX)*0.7):]), np.array(yy[int(len(yy)*0.7):])
```
```python=
# build many-to-many rnn model
def buildManyToManyModel(shape):
    model = Sequential()
    model.add(LSTM(13, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Lambda(lambda x: x[:, -5*24:, :]))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model
```
```python=
# read file and process data
from keras.layers import Lambda
from sklearn.model_selection import ShuffleSplit
df = readTrain('data.csv', [])
X_train, y_train, X_test, y_test = buildTrain(df, 5*24, 5*24)
splits = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
```
```python=
# training process
for train_idx, val_idx in splits.split(X_train):
    X_train_cv = X_train[train_idx]
    y_train_cv = y_train[train_idx]
    X_valid_cv = X_train[val_idx]
    y_valid_cv = y_train[val_idx]
    y_train_cv = y_train_cv[:,:,np.newaxis]
    y_valid_cv = y_valid_cv[:,:,np.newaxis]
    model = buildManyToManyModel(X_train_cv.shape)
    callback = EarlyStopping(monitor="val_loss", min_delta=0.1, patience=10, verbose=10, mode="auto")
    history = model.fit(X_train_cv, y_train_cv, epochs=500, batch_size=128, validation_data=(X_valid_cv, y_valid_cv), callbacks=[callback])
```
Final epoch:
loss: 539.6060 - val_loss: 532.8691
```python=
# plot the training curve
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(history.history['loss'],'b')
plt.plot(history.history['val_loss'],'r')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```
![](https://i.imgur.com/H2JgaYt.png)
It shows that the training and validation loss are close.
```python=
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
predictions = model.predict(X_test)
print(mean_squared_error(y_test,predictions[:,:,0]))
print(r2_score(y_test,predictions[:,:,0]))
print(explained_variance_score(y_test,predictions[:,:,0]))
```
output:
543.9387292990373
-0.007518498144947196
-7.342700505384251e-06
## Discussion
* From the metrics, we can see the result is not very good. There may be some reasons: (1) Features are not enough to predict the AQI (2) Predict the next 5 day AQI from previous 5 days may not be best way since the period is too long for my model to learn more information. Shortening the period and predict until 5 days by the predicted value may also be considered. However, in this way, the bias would increase along with the number of predicted data contained in the input data.
* Despite capability of predict accurately, my model is trained properly since the training, validation, and test loss are very close.