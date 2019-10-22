# beijing-house-scikit-learn
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
</br>

<pre><a href="https://jungh0.github.io/beijing-house-scikit-learn/map.html">https://jungh0.github.io/beijing-house-scikit-learn/map.html</a></pre>


## Basic concept of project

### Predict Housing Price
Housing Price of Beijing from 2011 to 2017, fetching from &nbsp; &nbsp;<img alt="icon" src="https://user-images.githubusercontent.com/8678595/67253579-3fe7f400-f4b3-11e9-90df-439e7e3620d0.png" width="100px">

### Descriptions for data
Dataset Size : 134266 <br>
![image](https://user-images.githubusercontent.com/8678595/67253643-bab10f00-f4b3-11e9-98dd-011d17b9767b.png)
![image](https://user-images.githubusercontent.com/8678595/67253645-bbe23c00-f4b3-11e9-9c40-2150ac057344.png)

## Data Preprocessing

### Data Value Changes (Cleaning dirty data)
![image](https://user-images.githubusercontent.com/8678595/67253661-de745500-f4b3-11e9-9770-d5e7ed1f296a.png)

``` python
#clean value
data = clean_value(data,'floor')
data = clean_value(data,'constructionTime')

#drop nan data
data_dnan = dropnan(data)
```

### Data Value Changes (Text preprocessing)
Remove the meaningless words <br>
Left only the floor number <br>

![image](https://user-images.githubusercontent.com/8678595/67253774-7a9e5c00-f4b4-11e9-8f72-e9035232720c.png)
![image](https://user-images.githubusercontent.com/8678595/67253775-7c681f80-f4b4-11e9-801c-17e772831f54.png)
![image](https://user-images.githubusercontent.com/8678595/67253777-7d994c80-f4b4-11e9-9172-8161d82b7fe8.png)


``` python
def clean_value(D,str):
    floor = D[str].str.extract('(\d+)')
    floor.columns = [str + '_new']
    D = drop(D,str) #drop original floor data
    D = concat(D,floor) #concat clean_floor to dataset
    return D
```

### Feature Engineering
Categorical data <br>
![image](https://user-images.githubusercontent.com/8678595/67253761-6b1f1300-f4b4-11e9-91f1-a92cec8ba80f.png)
![image](https://user-images.githubusercontent.com/8678595/67253764-6ce8d680-f4b4-11e9-966c-e53c16716e6f.png)
![image](https://user-images.githubusercontent.com/8678595/67253767-6e1a0380-f4b4-11e9-91c0-bd1a69cc205c.png)

``` python
def one_hot_data(D,c_str):
    D[c_str] = D[c_str].astype(str)
    one_hot_data = pd.get_dummies(D[[c_str]])
    D = drop(D,c_str)
    new = concat(D,one_hot_data)
    return new
```

### Preprocessing Results
After Preprocessing <br>
Dataset Size : 127670 <br>

![image](https://user-images.githubusercontent.com/8678595/67253826-be916100-f4b4-11e9-8103-e31c43622010.png)
![image](https://user-images.githubusercontent.com/8678595/67253833-bfc28e00-f4b4-11e9-9872-6ef17ebf7275.png)
![image](https://user-images.githubusercontent.com/8678595/67253839-ca7d2300-f4b4-11e9-8f93-edd9122fe6df.png)

## Data Analysis

### Find model
Use 102137 train set and 25534 test set<br>
Because the MSE is the smallest<br>
Use the Random Forest Regressor<br>

![image](https://user-images.githubusercontent.com/8678595/67253862-eed8ff80-f4b4-11e9-84f5-11d3625a3f18.png)

### Random Forest Regressor

#### Find estimators

``` python
def find_estimators(train_set_x,train_set_y,test_set_x,test_set_y):
    #MSE의 변화를 확인하기 위하여 앙상블의 크기 범위에서 랜덤 포레스트 트레이닝
    mseOos = []
    nTreeList = range(10, 500, 100)
    for iTrees in nTreeList:
        depth = None
        maxFeat = None #4 #조정해볼 것
        wineRFModel = RandomForestRegressor(n_estimators=iTrees,max_depth=None, max_features=maxFeat,oob_score=False, random_state=531)
        wineRFModel.fit(train_set_x, train_set_y)
        #데이터 세트에 대한 MSE 누적
        prediction = wineRFModel.predict(test_set_x)
        mseOos.append(mean_squared_error(test_set_y, prediction))
    print("MSE")
    print(mseOos[-1])
    #트레이닝 테스트 오차 대비  앙상블의 트리 개수 도표 그리기
    plt.plot(nTreeList, mseOos)
    plt.xlabel('Number of Trees in Ensemble')
    plt.ylabel('Mean Squared Error')
    plt.show()
```

Set estimator = 100 <br>
![image](https://user-images.githubusercontent.com/8678595/67253914-38c1e580-f4b5-11e9-99e9-acd9d4d60ba6.png)
![image](https://user-images.githubusercontent.com/8678595/67253916-3a8ba900-f4b5-11e9-8a21-94d56a57db62.png)

#### Find max depth

``` python
def find_max_depth(train_set_x,train_set_y,test_set_x,test_set_y):
    #MSE의 변화를 확인하기 위하여 앙상블의 크기 범위에서 랜덤 포레스트 트레이닝
    mseOos = []
    nTreeList = range(10, 50, 1)
    for iTrees in nTreeList:
        maxFeat = None
        wineRFModel = RandomForestRegressor(n_estimators=30,max_depth=iTrees, max_features=maxFeat,oob_score=False, random_state=531)
        wineRFModel.fit(train_set_x, train_set_y)
        #데이터 세트에 대한 MSE 누적
        prediction = wineRFModel.predict(test_set_x)
        mseOos.append(mean_squared_error(test_set_y, prediction))
    print("MSE")
    print(mseOos[-1])
    #트레이닝 테스트 오차 대비  앙상블의 트리 개수 도표 그리기
    plt.plot(nTreeList, mseOos)
    plt.xlabel('Number of maxFeat in Ensemble')
    plt.ylabel('Mean Squared Error')
    plt.show()
```

![image](https://user-images.githubusercontent.com/8678595/67253968-76bf0980-f4b5-11e9-8718-fa91c4cc1436.png)

#### Find max features

``` python
def find_max_features(train_set_x,train_set_y,test_set_x,test_set_y):
    #MSE의 변화를 확인하기 위하여 앙상블의 크기 범위에서 랜덤 포레스트 트레이닝
    mseOos = []
    nTreeList = range(1, 23, 1)
    for iTrees in nTreeList:
        wineRFModel = RandomForestRegressor(n_estimators=30,max_depth=17, max_features=iTrees,oob_score=False, random_state=531)
        wineRFModel.fit(train_set_x, train_set_y)
        #데이터 세트에 대한 MSE 누적
        prediction = wineRFModel.predict(test_set_x)
        mseOos.append(mean_squared_error(test_set_y, prediction))
    print("MSE")
    print(mseOos[-1])
    #트레이닝 테스트 오차 대비  앙상블의 트리 개수 도표 그리기
    plt.plot(nTreeList, mseOos)
    plt.xlabel('Number of max_features in Ensemble')
    plt.ylabel('Mean Squared Error')
    plt.show()
```

![image](https://user-images.githubusercontent.com/8678595/67253994-948c6e80-f4b5-11e9-8c57-d8063b027989.png)

## Data Analysis

### Result

![image](https://user-images.githubusercontent.com/8678595/67254010-b128a680-f4b5-11e9-8b50-a6fb3780605f.png)
![image](https://user-images.githubusercontent.com/8678595/67253764-6ce8d680-f4b4-11e9-966c-e53c16716e6f.png)
![image](https://user-images.githubusercontent.com/8678595/67254012-b2f26a00-f4b5-11e9-86ee-76097e0580fc.png)

### Feature Importance

![image](https://user-images.githubusercontent.com/8678595/67254039-d4535600-f4b5-11e9-9919-0018d2aec6e9.png)

### Predict housing price

![image](https://user-images.githubusercontent.com/8678595/67254049-de755480-f4b5-11e9-8943-208dc7f117c0.png)
![image](https://user-images.githubusercontent.com/8678595/67254050-dfa68180-f4b5-11e9-803a-42671c86db0c.png)
![image](https://user-images.githubusercontent.com/8678595/67254063-f64cd880-f4b5-11e9-9274-30677c661df2.png)
![image](https://user-images.githubusercontent.com/8678595/67254065-f77e0580-f4b5-11e9-9e89-da350a74c040.png)
<img src='https://github.com/jungh0/beijing-house-scikit-learn/blob/master/img/map_price.png' height='300px'/>
<img src='https://github.com/jungh0/beijing-house-scikit-learn/blob/master/img/map_price_zoom.png' height='300px'/>

