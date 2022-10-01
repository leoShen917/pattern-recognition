import numpy as np
import pandas as pd
def read_data(filename):
    dataset=pd.read_csv(filename)
    data=dataset.iloc[:,1:5]
    category = pd.Categorical(dataset.iloc[:,5])
    label=category.codes
    data=np.array(data.values.tolist())
    data=np.insert(data,0,1,axis=1)
    return data,label

def split_train(filename):
    data,label=read_data(filename)
    train_data=np.vstack((data[0:30],data[50:80],data[100:130]))
    train_label=np.append(label[0:30],label[50:80])
    train_label=np.append(train_label,label[100:130])
    return train_data,train_label

def split_test(filename):
    data,label=read_data(filename)
    test_data=np.vstack((data[30:50],data[80:100],data[130:150]))
    test_label=np.append(label[30:50],label[80:100])
    test_label=np.append(test_label,label[130:150])
    return test_data,test_label