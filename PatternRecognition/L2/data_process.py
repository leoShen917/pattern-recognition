import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def generate_data(number):                                     #生成数据并将其存到csv文件中
    if number==1:                                              #如果number=1，则生成线性可分的数据集
        mean1=[-5,0]
        mean2=[0,5] 
    if number==2:                                              #如果number=2，则生成线性不可分的数据集
        mean1=[1,0]
        mean2=[0,1]
    cov = [[1,0],[0,1]]
    datax1=np.random.multivariate_normal(mean1,cov,200)
    datax2=np.random.multivariate_normal(mean2,cov,200)
    x1=np.insert(datax1,0,1,axis=1)                            #增广后的数据集x1
    x2=np.insert(datax2,0,1,axis=1)                            #增广后的数据集x2
    x1_label=np.ones(200)
    x2_label=-np.ones(200)  
    all_data = np.vstack((x1,x2))
    all_label=np.append(x1_label,x2_label)
    data=pd.DataFrame(all_data,columns=["x0","x1","x2"])
    data['label']=all_label
    data.to_csv('data%d.csv'%number,index=0)                   #将相应数据集存入对应文件

def read_data(num):                                            #利用pd读取csv文件中的数据
    data=pd.read_csv("data%d.csv" %(num))
    all_data=data.iloc[:,0:3]
    all_data=all_data.values.tolist()
    all_label=list(data.iloc[:,3])
    all_data=np.array(all_data)
    all_label=np.array(all_label)
    return all_data,all_label

def split_train(num):                                           #提取出训练集数据
    all_data,all_label=read_data(num)
    train_data = np.vstack((all_data[:160],all_data[200:360]))
    train_label = np.append(all_label[:160],all_label[200:360])
    return train_data,train_label

def split_test(num):                                            #提取出测试集数据
    all_data,all_label=read_data(num)
    test_data = np.vstack((all_data[160:200],all_data[360:400]))
    test_label = np.append(all_label[160:200],all_label[360:400])
    return test_data,test_label

def draw(data,label,w,flag):                                    #画出分界面以及各样本点
    length=len(data)
    if flag==0:
        for i in range(length):
            if label[i]==1:
                plt.scatter(data[i][1],data[i][-1],c='r',marker='o')
            if label[i]==-1:
                plt.scatter(data[i][1],data[i][-1],c='b',marker='x')
    
    if flag==1:
        for i in range(length):
            if label[i]==1 and i<160:
                plt.scatter(data[i][1],data[i][-1],c='r',marker='o')
            if label[i]==1 and i>=160:
                plt.scatter(data[i][1],data[i][-1],c='m',marker='o')
            if label[i]==-1 and i<360:
                plt.scatter(data[i][1],data[i][-1],c='b',marker='x')
            if label[i]==-1 and i>=360:
                plt.scatter(data[i][1],data[i][-1],c='c',marker='x')

    if w[2]!=0:
        x=np.linspace(min(data[:,1])-1,max(data[:,1])+1,100)
        y=-w[1]/w[2]*x-w[0]/w[2]
        plt.plot(x,y,c='g')
    plt.show()
