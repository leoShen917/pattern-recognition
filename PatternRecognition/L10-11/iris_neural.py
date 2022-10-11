import torch 
from torch import nn
from torch.nn import init
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(4, 16) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(16,3)  # 输出层
    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

def softmax(X):                                 #进行softmax处理
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

def cross_entropy(y_hat, y):                    #计算交叉熵
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))

def train_iris(data,label,t_data,t_label,net,epoch,alpha):     #训练神经网络
    L=np.zeros(epoch)
    ac=np.zeros(epoch)
    for name, param in net.named_parameters():
        if name.startswith("weight"):
    	    nn.init.xavier_uniform_(param,gain=1)
    optimizer = torch.optim.Adam(net.parameters(),lr=alpha) #设置梯度下降
    for t in range(epoch):
        optimizer.zero_grad()
        y_hat=softmax(net(data))                #求出每个样本分到各标签的概率
        l = cross_entropy(y_hat, label).sum()/90
        L[t] = cross_entropy(y_hat, label).sum()/90   #计算交叉熵损失函数
        l.backward()                            #求梯度
        optimizer.step()
        test_hat=softmax(net(t_data))
        ac[t]=accuracy(test_hat,t_label)
    return L,ac

def accuracy(y_hat, y):                #计算分类正确率
        return (y_hat.argmax(dim=1) == y).float().mean().item()
        
def test_iris(data,label,net):
    y_hat=softmax(net(data))
    print(accuracy(y_hat,label))

if __name__ == '__main__':
    epoch=5000
    alpha=0.001
    iris=load_iris()
    iris_d=pd.DataFrame(iris['data'],columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'])
    data=torch.tensor(iris['data'])
    label=torch.tensor(iris['target'])
    data=data.to(torch.float32)
    data1=data[0:50]
    data2=data[50:100]
    data3=data[100:150]
    label=label.to(torch.int64)
    label1=label[0:50]
    label2=label[50:100]
    label3=label[100:150]
    train_data1,test_data1,train_label1,test_label1=model_selection.train_test_split(data1,label1,random_state=0,test_size=0.4)
    train_data2,test_data2,train_label2,test_label2=model_selection.train_test_split(data2,label2,random_state=1,test_size=0.4)
    train_data3,test_data3,train_label3,test_label3=model_selection.train_test_split(data3,label3,random_state=2,test_size=0.4)
    train_data=torch.cat((train_data1,train_data2))
    train_data=torch.cat((train_data,train_data3))
    test_data=torch.cat((test_data1,test_data2))
    test_data=torch.cat((test_data,test_data3))
    train_label=torch.cat((train_label1,train_label2))
    train_label=torch.cat((train_label,train_label3))
    test_label=torch.cat((test_label1,test_label2))
    test_label=torch.cat((test_label,test_label3))
    L=np.zeros((5,epoch))
    ac=np.zeros((5,epoch))
    t=np.arange(0,5000)
    net=nn.Sequential(nn.Linear(4, 16),nn.ReLU(),nn.Linear(16, 3))
    L[0],ac[0]=train_iris(train_data,train_label,test_data,test_label,net,epoch,alpha)
    net=nn.Sequential(nn.Linear(4, 16),nn.Sigmoid(),nn.Linear(16, 3))
    L[1],ac[1]=train_iris(train_data,train_label,test_data,test_label,net,epoch,alpha)
    net=nn.Sequential(nn.Linear(4, 16),nn.Tanh(),nn.Linear(16, 3))
    L[2],ac[2]=train_iris(train_data,train_label,test_data,test_label,net,epoch,alpha)
    net=nn.Sequential(nn.Linear(4, 16),nn.LeakyReLU(),nn.Linear(16, 3))
    L[3],ac[3]=train_iris(train_data,train_label,test_data,test_label,net,epoch,alpha)
    net=nn.Sequential(nn.Linear(4, 16),nn.ReLU(),nn.Linear(16, 3))
    L[4],ac[4]=train_iris(train_data,train_label,test_data,test_label,net,epoch,alpha)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.plot(t, L[0], label='ReLU')
    plt.plot(t, L[1], label='Sigmoid')
    plt.plot(t, L[2], label='Tanh')
    plt.plot(t, L[3], label='LeakyReLU')
    
    plt.legend()
    plt.show()
    plt.plot(t, ac[0], label='ReLU')
    plt.plot(t, ac[1], label='Sigmoid')
    plt.plot(t, ac[2], label='Tanh')
    plt.plot(t, ac[3], label='LeakyReLU')
    
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('test_accuracy')
    plt.show()