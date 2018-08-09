import os
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import r2_score


def main():
    path = "F:\BigData\DataScience\DataScience-Python3\emails\ham"
    path1 = "F:\BigData\DataScience\DataScience-Python3\emails\spam"
    classify=["ham","spam"]
    final_ham=pd.DataFrame()
    #if(final_ham.empty==True):
    final_ham=createDataFrame(path,classify[0])
    final_ham=final_ham.append(createDataFrame(path1,classify[1]))
    train_ham,test_ham=train_test_split(final_ham,test_size=0.2)
    vectorizer=CountVectorizer()
    model=MultinomialNB()
    counts=vectorizer.fit_transform(train_ham['Message'].values)
    val=train_ham['class'].values
    model.fit(counts,val)
    test_counts=vectorizer.transform(test_ham['Message'])
    prediction=model.predict(test_counts)
    final=pd.DataFrame({'Message':test_ham['Message'],'class':test_ham['class'],'Predicted':prediction})
    final['Compare']=final['class']==final['Predicted']
    print(final.groupby('Compare').count())



def read(path):
    for file in os.listdir(path):
        path_red=os.path.join(path,file)
        inBody = False
        lines = []
        f = open(path_red, 'r',encoding='latin1')
        for line in f:
           # print (line)
            if inBody:
                lines.append(line)
            elif line == '\n':
                inBody = True
        f.close()
        message = '\n'.join(lines)
        yield (path,message)

def createDataFrame(path,classify):
    rows=[]
    index=[]
    for name,message in read(path):
        rows.append({"Message":message, "class":classify})
        index.append(name)
    return (pd.DataFrame(rows,index=index))


if __name__=="__main__" :
    main()

