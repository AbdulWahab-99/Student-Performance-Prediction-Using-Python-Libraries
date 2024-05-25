import pandas as pd

df=pd.read_csv('StudentsPerformance.csv')
print(df.head())
print(df.tail())
print(df.describe())
print(df.info())
df['avarage score']=df[['math score','reading score','writing score']].mean(axis=1)
categorical=df.drop(['math score','reading score', 'writing score','avarage score'], axis=1)
numerical=df[['math score','reading score', 'writing score','avarage score']]
df1= categorical.apply(lambda x: pd.factorize(x)[0])
data=pd.concat([df1,numerical],axis=1,ignore_index=True)
new_columns_name={0:'gender',1:'race',2:'parent education',3:'lunch',4:'preparetion tests',5:'math',6:'reading',7:'writing',8:'avarage'}
data=data.rename(columns=new_columns_name)
x=data.drop(['avarage'],axis=1)
y=data['avarage']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lr= LinearRegression()
model=lr.fit(x_train, y_train)
predict = model.predict(x_test)