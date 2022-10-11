from matplotlib.pyplot import legend
import numpy as np
import data_process as dp

def sigmoid(x):
    return 1/(1+np.exp(-x))

def Loss(data,label,w):
    Loss=0
    N=len(data[0])
    for i in range(N):
        Loss+=np.log(1+np.exp(-label[i]*np.dot(w,data[i].T)))
    return Loss/N

def gradient(data,label,w):
    L=np.zeros(len(data[0]))
    for i in range(len(data)):
        L+=sigmoid(-label[i]*w*data[i].T)*(-label[i]*data[i])
    return L/len(data)    

def Logistic(data,label,alpha,epoch):               #Logistic回归
    datadim=len(data[0])
    w=np.zeros(datadim)
    lost=np.zeros(epoch)                            
    ep=np.zeros(epoch)
    for t in range(epoch):
        L=gradient(data,label,w)                    #求梯度
        w-=alpha*L                                  #权重更新
        ep[t]=t
        lost[t]=Loss(train_data,train_label,w)      #求出损失函数
        if(L==np.zeros(datadim)).all():
            break
    dp.epoch_line(lost,ep)                          #画出损失函数同迭代次数的变化
    return w

def predict(all_data,all_label,testdata,w):         #分类器在测试集样本属"1"类的概率值
    length=len(testdata)
    probability=np.zeros(length)
    y=np.ones(length)
    dp.draw(all_data,all_label,w)                   #画出分类面和数据集
    for i in range(length):
        probability[i] = sigmoid(-np.dot(w,test_data[i]))
        if probability[i]<0.5:                      #若概率小于0.5，则属“-1”类
            y[i]=-1
    return probability,y    


if __name__ == '__main__':
    num=1                                                     #不同的num对应读取不同的数据集，num=1为线性可分数据集，num=2为线性不可分数据集
    all_data,all_label=dp.read_data(num)                      #提取数据集
    train_data,train_label=dp.split_train(num)                #训练集
    test_data,test_label=dp.split_test(num)                   #测试集
    w=Logistic(train_data,train_label,0.01,1000)              #logistic回归
    probability,predict_label=predict(all_data,all_label,test_data,w)  #测试集上概率值和预测标签