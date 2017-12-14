
# coding: utf-8

# In[1]:


import csv

def transformDataTitanic(trainingFile, features):
    transformData=[]
    labels=[]
    
    gendermap={"male":1,"female":2,"":""}
    embarkMap={"C":1,"Q":2,"S":3,"":""}
    blank=""
    
    with open(trainingFile,'r') as csvfile:
        lineReader=csv.reader(csvfile,delimiter=",",quotechar="\"")
        linenum=1
        for row in lineReader:
            if(linenum==1):
                header=row
            else:
                allFeatures=list(map(lambda x: gendermap[x] if row.index(x)==4 else embarkMap[x] if row.index(x)==11 else x, row))
                featureVector=[allFeatures[header.index(feature)] for feature in features]
                
                if blank not in featureVector:
                    transformData.append(featureVector)
                    labels.append(int(row[1]))
            linenum+=1
    return transformData,labels


# In[2]:


trainingFile="train.csv"
features=["Pclass","Sex","Age","SibSp","Fare","Parch","Embarked"]
trainingData=transformDataTitanic(trainingFile,features)


# In[3]:


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
clf=RandomForestClassifier(n_estimators=10)
X=np.array(trainingData[0])
Y=np.array(trainingData[1])
clf=clf.fit(X,Y)


# In[4]:


def transformTestDataTitanic(testFile,features):
    transformData=[]
    ids=[]
    genderMap={"male":1,"female":2,"":""}
    embarkMap={"C":1,"Q":2,"S":3,"":""}
    blank=""
    with open(testFile,"r") as csvfile:
        lineReader=csv.reader(csvfile,delimiter=",",quotechar="\"")
        linenum=1
        for row in lineReader:
            if(linenum==1):
                header=row
            else:
                allFeatures=list(map(lambda x:genderMap[x] if row.index(x)==3 else embarkMap[x] if row.index(x)==10 else x, row))
                # The second column is Passenger class, let the default value be 2nd class
                if allFeatures[1]=="":
                    allFeatures[1]=2
                # Let the default age be 30
                if allFeatures[4]=="":
                    allFeatures[4]=30
                # Let the default number of companions be 0 (assume if we have no info, the passenger
                # was travelling alone)
                if allFeatures[5]=="":
                    allFeatures[5]=0
                # By eyeballing the data , the average fare seems to be around 30
                if allFeatures[8]=="":
                    allFeatures[8]=32
                featureVector=[allFeatures[header.index(feature)] for feature in features]
                #featureVector=list(map(lambda x:0 if x=="" else x, featureVector))
                transformData.append(featureVector)
                ids.append(row[0])
            linenum=linenum+1 
    return transformData,ids


# In[5]:


def titanicTest(classifier,resultFile,transformDataFunction=transformTestDataTitanic):
    testFile="test.csv"
    testData=transformDataFunction(testFile,features)
    result=classifier.predict(testData[0])
    with open(resultFile,"w") as f:
        ids=testData[1]
        lineWriter=csv.writer(f,delimiter=',',quotechar="\"")
        lineWriter.writerow(["PassengerId","Survived"])
        for rowNum in range(len(ids)):
            try:
                lineWriter.writerow([ids[rowNum],result[rowNum]])
            except(Exception,e):
                print(e)
resultFile="result4.csv"
titanicTest(clf,resultFile)

