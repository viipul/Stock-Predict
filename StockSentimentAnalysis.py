import pandas as pd

data=pd.read_csv('Data.csv',encoding='ISO-8859-1')
print(data.head())
train=data[data['Date']<'20150101']
test=data[data['Date']>'20141231']

#removing pancuation
data1=train.iloc[:,2:27]
data1.replace('[^a-zA-z]',' ',regex=True,inplace=True)

#renaming columns names for ease of access
list1=[i for i in range(0,25)]
new_index=[str(i) for i in list1]
data1.columns=new_index
print(data1)

for i in new_index:
    data1[i]=data1[i].str.lower()
print(data1.head(1))

#combining all the headlines for a particular row into a paragraph
headlines=[]
for i in range(0,len(data1.index)):
    headlines.append(''.join(str(j) for j in data1.iloc[i,0:25]))
print(headlines[0])

#Applying classification model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

#IMPLEMENTING BAG OF WORDS
print('*********IMPLEMENTING BAG OF WORDS************')
count_vector=CountVectorizer(ngram_range=(2,2))
train_data=count_vector.fit_transform(headlines)
print('BAG OF WORDS APPLIED')

#IMPLEMENTING BAG OF WORDS
print('*********RANDOM FOREST CLASSIFIER************')
random_f=RandomForestClassifier(n_estimators=200,criterion='entropy')
random_f.fit(train_data,train['Label'])
print('RANDOM FOREST APPLIED')
#Prediction from test data
#test data transform
test_transform=[]
for i in range(0,len(test.index)):
    test_transform.append(' '.join(str(j) for j in test.iloc[i,2:27]))
test_dataset=count_vector.transform(test_transform)
predictions=random_f.predict(test_dataset)

#checking accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix=confusion_matrix(test['Label'],predictions)
print('Confusion matrix')
print(matrix)
score=accuracy_score(test['Label'],predictions)
print('Accuracy Score')
print(score)
report=classification_report(test['Label'],predictions)
print('Classification Report')
print(report)