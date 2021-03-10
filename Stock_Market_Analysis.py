import pandas as pd
import pickle
df=pd.read_csv("Data_For_Stocks.csv")
#print(df.head())
train=df[df['Date'] < '20150101']
test=df[df['Date'] > '20141231']
data= train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ", regex=True, inplace= True)
#print(data.head())
list1=[i for i in range(25)]
new_index=[str(i) for i in list1]
data.columns=new_index

for index in new_index:
	data[index]= data[index].str.lower()
#print(data.head())

headlines=[]
for x in range(0,len(data.index)):
	headlines.append(' '.join(str(y) for y in data.iloc[x,0:25]))
#print(headlines[0])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
#print("Successfully Imported")
countvector=CountVectorizer(ngram_range=(2,2))
trainDataset=countvector.fit_transform(headlines)
#print(trainDataset[0])

randomclassifier=RandomForestClassifier(n_estimators=200, criterion='entropy')
randomclassifier.fit(trainDataset,train['Label'])

test_transform=[]
for x in range(0,len(test.index)):
	test_transform.append(' '.join(str(x) for x in test.iloc[x,2:27]))
test_dataset=countvector.transform(test_transform)
predictions=randomclassifier.predict(test_dataset)
#print(predictions)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
matrix=confusion_matrix(test['Label'], predictions)
print(matrix)
score=accuracy_score(test['Label'], predictions)
print(score)
report=classification_report(test['Label'], predictions)
print(report)