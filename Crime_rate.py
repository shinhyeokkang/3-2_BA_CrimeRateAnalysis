import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor

train15=pd.read_csv(r'C:\data2015.csv')#파일명에 한글이 있으면 에러남
train16=pd.read_csv(r'C:\data2016.csv')#그래서 engine='python' 설정해주니 읽어지는데 한글 깨짐
train17=pd.read_csv(r'C:\data2017.csv')#그냥 파일명 영어로 만드는게 답
test=pd.read_csv(r'C:\data2018.csv')

train=pd.concat([train15,train16,train17])
train=train.set_index(train['district'])
train.drop('district',axis=1,inplace=True)
test=test.set_index(test['district'])
test.drop('district',axis=1,inplace=True)
corrtrain = train.corr()

'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
train = pd.DataFrame(train)

name = 
train.colulmns = ['women scouts performance', 'unauthorized building', 'CCTV POP',
       'local tax payment', 'number of bars', 'average age', 'population', 
       'education', 'fine dust', 'single family', 'traditional market', 'cultural area',
       'murder', 'burglar', 'rape', 'theft', 'violence', 'total of 5crime']
train
'''

train.columns
varlist = ['women scouts performance', 'unauthorized building',
       'local tax payment', 'number of bars', 'average age', 'population', 
       'education', 'fine dust', 'single family']
#강간
y = ['murder', 'burglar', 'rape', 'theft', 'violence', 'total of 5crime']
qq = ['women scouts performance', 'unauthorized building', 'CCTV POP', 
      'fine dust', 'single family','traditional market', 'cultural area',
      'murder', 'burglar', 'rape', 'theft', 'violence', 'total of 5crime']

trainy = train[['murder', 'burglar', 'rape', 'theft', 'violence', 'total of 5crime']]
train.drop(y,axis=1,inplace=True)
trainx = train
testy = test[y]
test.drop(y,axis=1,inplace=True)
testx = test
testY=testy[y[i]]
#total of 5crime 예측
#LinearRegrssion
i=1
reg=LinearRegression()
reg.fit(trainx,trainy[y[i]])
testY=testy[y[i]]
pred=reg.predict(testx)
print("burgler")
print(reg.score(testx,testY))

train15=pd.read_csv(r'C:\data2015.csv')#파일명에 한글이 있으면 에러남
train16=pd.read_csv(r'C:\data2016.csv')#그래서 engine='python' 설정해주니 읽어지는데 한글 깨짐
train17=pd.read_csv(r'C:\data2017.csv')#그냥 파일명 영어로 만드는게 답
test=pd.read_csv(r'C:\data2018.csv')

train=pd.concat([train15,train16,train17])
train=train.set_index(train['district'])
train.drop('district',axis=1,inplace=True)
test=test.set_index(test['district'])
test.drop('district',axis=1,inplace=True)
corrtrain = train.corr()
#절도
'''
y = ['murder', 'burglar', 'rape', 'theft', 'violence', 'total of 5crime']
qq = ['unauthorized building', 
      'fine dust','single family', 'traditional market', 'cultural area',
      'murder', 'burglar', 'rape', 'theft', 'violence', 'total of 5crime']
'''
trainy = train[['murder', 'burglar', 'rape', 'theft', 'violence', 'total of 5crime']]
train.drop(qq,axis=1,inplace=True)
trainx = train
testy = test[y]
test.drop(qq,axis=1,inplace=True)
testx = test

#total of 5crime 예측
#LinearRegrssion
i=3
reg=LinearRegression()
reg.fit(trainx,trainy[y[i]])
testY=testy[y[i]]
pred=reg.predict(testx)
print("절도")
print(reg.score(testx,testY))


#
train15=pd.read_csv(r'C:\data2015.csv')#파일명에 한글이 있으면 에러남
train16=pd.read_csv(r'C:\data2016.csv')#그래서 engine='python' 설정해주니 읽어지는데 한글 깨짐
train17=pd.read_csv(r'C:\data2017.csv')#그냥 파일명 영어로 만드는게 답
test=pd.read_csv(r'C:\data2018.csv')

train=pd.concat([train15,train16,train17])
train=train.set_index(train['district'])
train.drop('district',axis=1,inplace=True)
test=test.set_index(test['district'])
test.drop('district',axis=1,inplace=True)
corrtrain = train.corr()
#폭력
'''
y = ['murder', 'burglar', 'rape', 'theft', 'violence', 'total of 5crime']
qq = ['unauthorized building', 
      'single family', 'traditional market',
      'murder', 'burglar', 'rape', 'theft', 'violence', 'total of 5crime']
'''
trainy = train[['murder', 'burglar', 'rape', 'theft', 'violence', 'total of 5crime']]
train.drop(qq,axis=1,inplace=True)
trainx = train
testy = test[y]
test.drop(qq,axis=1,inplace=True)
testx = test

#total of 5crime 예측
#LinearRegrssion
i=4
reg=LinearRegression()
reg.fit(trainx,trainy[y[i]])
testY=testy[y[i]]
pred=reg.predict(testx)
print("폭력")
print(reg.score(testx,testY))


corrtest = test.corr()
'''
#heatmap
plt.figure(figsize=(10,10))
sns.heatmap(data = corrtrain, annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')

plt.figure(figsize=(10,10))
sns.heatmap(data = corrtest, annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')

#pairplot
sns.pairplot(train[['women scouts performance', 'unauthorized building', 'CCTV POP',
       'local tax payment', 'number of bars', 'average age', 'population', 'education',
       'fine dust', 'single family', 'traditional market', 'cultural area']])
sns.pairplot(test[['women scouts performance', 'unauthorized building', 'CCTV POP',
       'local tax payment', 'number of bars', 'average age', 'population', 'education',
       'fine dust', 'single family', 'traditional market', 'cultural area']])
#VIF
def cal_VIF(k):

    for i in range(k):
        reg=LinearRegression()
        X=trainx[np.setdiff1d(varlist,[varlist[i]])]
        y=trainx[varlist[i]]
        reg.fit(X,y)
        r_square=reg.score(X,y)
        vif = 1/(1-r_square)   
        print(varlist[i],vif)    
def cal_VIF18(k):

    for i in range(k):
        reg=LinearRegression()
        X=testx[np.setdiff1d(varlist,[varlist[i]])]
        y=testx[varlist[i]]
        reg.fit(X,y)
        r_square=reg.score(X,y)
        vif = 1/(1-r_square)   
        print(varlist[i],vif)
print("train")
cal_VIF(12)
print("2018년")
cal_VIF18(12)

'''

i=5
#K-nn
nn_reg=KNeighborsRegressor(n_neighbors=3)
nn_reg.fit(trainx,trainy[y[i]])
pred2=nn_reg.predict(testx)
nn_reg.score(testx,testY)
#Lasso
lasso=Lasso(alpha=1)
lasso.fit(trainx,trainy[y[i]])
pred3=lasso.predict(testx)
lasso.score(testx,testY)
#Ridge
ridge=Ridge(alpha=1.0)
ridge.fit(trainx,trainy[y[i]])
pred4=ridge.predict(testx)
ridge.score(testx,testY)
#ElasticNet
elsnet=ElasticNet(alpha=1, l1_ratio=0.7)
elsnet.fit(trainx,trainy[y[i]])
pred5=elsnet.predict(testx)
elsnet.score(testx,testY)


#plot among them
xx=np.linspace(0,max(testY),100)
fig=plt.figure(figsize=(10,8))
ax=plt.gca()
plt.scatter(testY,pred, label='Linear regression')
plt.scatter(testY,pred2, label='$k$-NN')
plt.scatter(testY,pred3, label='Lasso')
plt.scatter(testY,pred4, label='Ridge')
plt.scatter(testY,pred5, label='ElasticNet')
plt.plot(xx,xx)
plt.ylabel('Estimation',fontsize=16)
plt.xlabel('True output', fontsize=16)
plt.legend(fontsize=14)
ax.tick_params(labelsize=16)

print("LR")
reg.score(testx,testY)
print("K-nn")
nn_reg.score(testx,testY)
print("Lasso")
lasso.score(testx,testY)
print("Ridge")
ridge.score(testx,testY)
print("ElasticNet")
elsnet.score(testx,testY)

#cross-validation
from sklearn.model_selection import KFold
X=["a","b","c","d","e","f"]
kf = KFold(n_splits=3, shuffle=True, random_state=1)
for train,test in kf.split(train):
    print("%s %s" % (train,test))


reg1=LinearRegression()
reg1.fit(train,trainy)

reg1.score(train,trainy)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(reg1,train, trainy, cv=3) # model, train, target, cross validation


