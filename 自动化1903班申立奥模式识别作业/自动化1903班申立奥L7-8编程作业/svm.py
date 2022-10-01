from math import exp,ceil
from operator import length_hint
import numpy as np
import cvxopt
import random
from sklearn.preprocessing import PolynomialFeatures
import data_process as dp
def primal_SVM(data,label):                 #原数据是增广数据
    data=np.delete(data,0,1)                #去掉增广项
    d=len(data[0])
    length=len(data)
    Q1=np.zeros((1,d+1))
    Q2=np.hstack((np.zeros((1,d)).T,np.eye(d)))
    Q=cvxopt.matrix(np.vstack((Q1,Q2)))
    p=cvxopt.matrix(np.zeros((d+1,1)))
    A=np.zeros((length,d+1))
    for i in range(length):
        a=np.insert(data[i],0,1)
        A[i]=label[i]*a
    A=cvxopt.matrix(-A)
    c=cvxopt.matrix(-np.ones((length,1)))
    u=cvxopt.solvers.qp(Q,p,A,c)            #利用二次规划问题求解问题
    w=np.ravel(u['x'])                      #得到最佳分类面
    return w

def shuffle(data,label):                      #重洗数据
    index= [i for i in range(len(data))]
    random.shuffle(index)
    data=data[index]
    label=label[index]
    return data,label

def check(data,label,w):                      #检查是否所有数据都满足ywx>=1
    count=0
    for i in range(len(data)):
        if 1-label[i]*np.dot(w,data[i])>0:
            count += 1
    return count

def Hinge_loss(data,label,batch,epoch,alpha): #小批量随机梯度下降
    d=len(data[0])
    t=0
    length =len(data)
    w = np.zeros(d)
    while t<epoch:
        t+=1
        data,label=shuffle(data,label)        #重洗数据
        for count in  range(ceil(length/batch)):
            L = np.zeros(d)
            for i in range(batch):
                k=count*batch+i
                if k<length and 1-label[k]*np.dot(w,data[k])>0 :
                    L+=(-label[k]*data[k])/batch
            w-=alpha*L
        if check(data,label,w)==0:
            break
    return w 

def dual_SVM(data,label):
    data=np.delete(data,0,1)                          #去掉数据中的增广项
    length=len(data)
    dim=1                                             #升维到dim维
    poly = PolynomialFeatures(dim)
    Z=poly.fit_transform(data)                         
    YZ=np.delete(Z,0,axis=1)
    Z=np.delete(Z,0,axis=1)
    for i in range(length):
        if label[i]==-1:
            YZ[i]*=-1
    Q=cvxopt.matrix(np.dot(YZ,YZ.T))
    p=cvxopt.matrix(-np.ones((length,1)))
    A=cvxopt.matrix(-np.eye(length))
    c=cvxopt.matrix(np.zeros((length,1)))
    r=cvxopt.matrix(np.array([label]))
    v=cvxopt.matrix(np.zeros(1))
    u=cvxopt.solvers.qp(Q,p,A,c,r,v)                  #利用二次规划求解alpha
    alpha=np.ravel(u['x'])                            #获得拉普拉斯乘子
    
    sv=alpha>1e-5                                      
    alpha=alpha[sv]                                   #非零拉格朗日乘子对应的向量为支撑向量
    support=Z[sv]     
    print(support)
    support_label=label[sv]
    w=np.zeros(len(Z[0]))
    for i in range((len(alpha))):
        w+=alpha[i]*support_label[i]*support[i]       #最佳权系数w
    b=support_label[0]-np.dot(w,support[0])           #截距项
    w=np.insert(w,0,b)                                #增广后的w
    print(w)
    return w

class Kernel_SVM:                                     
    def __init__(self,flag,zeta,gamma,index):         #参数初始化
        self.flag=flag                                #flag=0,则为多项式核函数；flag=1,则为高斯核函数
        #多项式核函数参数
        self.zeta=zeta                                
        self.gamma=gamma
        self.index=index

    def kernel(self,x1,x2):                           #求核函数
        if self.flag==0:
            K=(self.zeta+self.gamma*np.dot(x1,x2))**self.index
        if self.flag==1:
            K=exp(-self.gamma*np.linalg.norm(x1-x2))
        return K

    def SVM(self,data,label):
        data=np.delete(data,0,1)
        length=len(data)
        K=np.zeros((length,length))
        for i in range(length):
            for j in range(length):
                K[i,j]=self.kernel(data[i],data[j])
        Q=cvxopt.matrix(np.outer(label,label)*K)
        p=cvxopt.matrix(-np.ones(length))
        A=cvxopt.matrix(-np.eye(length))
        c=cvxopt.matrix(np.zeros((length,1)))
        r=cvxopt.matrix(np.array([label]))
        v=cvxopt.matrix(np.zeros(1))
        u=cvxopt.solvers.qp(Q,p,A,c,r,v)              #利用二次规划求解alpha
        alpha=np.ravel(u['x'])                        #获得拉普拉斯乘子
        sv=alpha>1e-5
        self.alpha=alpha[sv]                          #非零拉格朗日乘子对应的向量为支撑向量
        self.support=data[sv]             
        print(self.support)
        self.support_label=label[sv]
        self.b=self.support_label[0]                  #截距项
        for i in range(len(self.alpha)):
            self.b-=alpha[i]*label[i]*self.kernel(self.support[i],self.support[0])

    def predict(self,data,label):                     #分类器在测试集中的准确率
        data=np.delete(data,0,1)
        count=0
        for n in range(len(data)):
            all=self.b
            for i in range(len(self.alpha)):
                all+=self.alpha[i]*self.support_label[i]*self.kernel(self.support[i],data[n])
            g=np.sign(all)
            if g==label[n]:
                count+=1
        precision=count/len(data)
        return precision

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
def train_pre(data,label,w):
    count=0
    length=len(data)
    y=np.zeros(length)
    for i in range(length):
        y[i] = np.sign(np.dot(w,data[i].T))
        if y[i]==label[i]:
            count+=1
    precision=count/length
    return precision


if __name__ == '__main__':
    num=2                                                     #不同的num对应读取不同的数据集，num=1为线性可分数据集，num=2为线性不可分数据集
    all_data,all_label=dp.read_data(num)                      #提取数据集
    train_data,train_label=dp.split_train(num)                #训练集
    test_data,test_label=dp.split_test(num)                   #测试集
    #w=Hinge_loss(data,label,1,100,1)
    #print(w)
    #w=primal_SVM(train_data,train_label)                     #Primal-SVM
    #dp.draw(train_data,train_label,w,1)
    dp.draw_high(train_data,train_label)
    #w=dual_SVM(train_data,train_label)                       #dual-SVM
    #print(w)
    #precision=predict(all_data,all_label,test_data,test_label,w)  #测试集正确率 
    #print(precision)
    #pre=train_pre(train_data,train_label,w)                  #训练集正确率
    #print(pre)
    #S=Kernel_SVM(1,1,1,4)                                    #Kernel-SVM
    #S.SVM(train_data,train_label)
    #precision=S.predict(test_data,test_label)
    #print(precision)                                         #测试集正确率

    #钓鱼岛问题
    #X_train, y_train, city, X_test, y_test=dp.data_generator()
    #dp.diaoyu(X_train, y_train, city, X_test, y_test)