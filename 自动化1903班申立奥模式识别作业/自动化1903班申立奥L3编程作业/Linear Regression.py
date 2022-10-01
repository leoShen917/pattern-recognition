import numpy as np
import data_process as dp
import time 
def generalized_inverse(train_data,train_label):    #广义逆求解析解
    X=train_data
    Y=train_label.T
    A=np.dot(X.T,X)
    Apinv=np.linalg.pinv(A)
    Xw=np.dot(Apinv,X.T)
    w=np.dot(Xw,Y)
    return w

def Lost(w,data,label):                             #求损失函数
    length=len(data)
    lost=0
    for i in range(length):
        lost+=(np.dot(w,data[i].T)-label[i])**2
    return lost/length

def gradient(data,label,w):                         #求梯度
    datadim=len(train_data[0])
    L=np.zeros(datadim)
    for i in range(len(data)):
        L=np.add(L,(np.dot(w,data[i].T)-label[i])*data[i])
    return L

def descent(train_data,train_label,alpha,epoch):    #梯度下降求最优解
    datadim=len(train_data[0])
    w=np.zeros(datadim)
    lost=np.zeros(epoch)
    ep=np.zeros(epoch)
    for t in range(epoch):                          #epoch为算法迭代次数
        L=gradient(train_data,train_label,w)        #更新w，学习率为alpha
        w =np.subtract(w,alpha*L)
        ep[t]=t
        lost[t]=Lost(w,train_data,train_label)
        if (L==np.zeros(datadim)).all():
            break;
    dp.epoch_line(lost,ep)
    return w

def predict(all_data,all_label,testdata,testlabel,w): #分类器在测试集中的准确率
    count=0
    length=len(testdata)
    y=np.zeros(length)
    dp.draw(all_data,all_label,w,1)                   #画出分类面和数据集
    for i in range(length):
        y[i] = np.sign(np.dot(w,testdata[i].T))
        if y[i]==testlabel[i]:
            count+=1
    precision=count/length
    return precision

if __name__ == '__main__':
    num=1                                                     #不同的num对应读取不同的数据集，num=1为线性可分数据集，num=2为线性不可分数据集
    all_data,all_label=dp.read_data(num)                      #提取数据集
    train_data,train_label=dp.split_train(num)                #训练集
    test_data,test_label=dp.split_test(num)                   #测试集
    start=time.perf_counter()                                 #计时
    #w=generalized_inverse(train_data,train_label)            #广义逆法
    w=descent(train_data,train_label,0.0001,100)             #梯度下降法
    precision=predict(all_data,all_label,test_data,test_label,w)  #测试集上预测样本正确率
    end=time.perf_counter()
    dur = end - start
    print(w)
    print(precision)
    print('Running time: %s Seconds'%(dur))
    

        