import torch 
from torch import nn
from torch.nn import init
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def read_data():                                #读取mnist文件
    mnist_train=torchvision.datasets.MNIST(root="",train=True,download=True,transform=transforms.ToTensor())
    mnist_test=torchvision.datasets.MNIST(root="",train=False,download=True,transform=transforms.ToTensor())
    batch_size = 256
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return train_iter,test_iter

def softmax(X):                                 #进行softmax处理
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5,stride=1,padding=2), 
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5,stride=1,padding=0),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

def epoch_plot(epoch,L,train_ac,test_ac):       #画出损失函数，训练集、测试集正确率与迭代次数的曲线
    t=np.linspace(0,epoch-1,num=epoch)
    plt.title('Variation of Loss function with epoch')
    plt.xlabel('epoch')
    plt.ylabel('L')
    plt.plot(t,L)
    plt.show()
    plt.title('Variation of train_accuracy function with epoch')
    plt.xlabel('epoch')
    plt.ylabel('train_accuracy')
    plt.plot(t,train_ac)
    plt.show()
    plt.title('Variation of test_accuracy function with epoch')
    plt.xlabel('epoch')
    plt.ylabel('test_accuracy')
    plt.plot(t,test_ac)
    plt.show()

def cross_entropy(y_hat, y):                    #计算交叉熵
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))

def accuracy(y_hat, y):                #计算分类正确率
    return (y_hat.argmax(dim=1) == y).float().mean().item()

def train_LeNet(train_iter,test_iter,net,epoch,alpha):  #训练LeNet神经网络
    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)
        params.requires_grad_(requires_grad=True)       #初始化网络参数
    ac=np.zeros(epoch)
    L=np.zeros(epoch)
    optimizer = torch.optim.Adam(net.parameters(), lr=alpha)    #采用adam法梯度下降
    test_ac=np.zeros(epoch)
    for t in range(epoch):
        count=0
        for X, y in train_iter:
            y_hat = softmax(net(X))                     #求出每个样本分到各标签的概率
            L[t]+=cross_entropy(y_hat, y).sum()
            ac[t]+=accuracy(y_hat,y)    
            l = cross_entropy(y_hat,y).sum()            #求出此batch内交叉熵损失函数
            l.backward()                                #反向传播
            optimizer.step()
            for params in net.parameters():
                params.grad.zero_()                     #梯度重置为0
            count+=1

        L[t]=L[t]/count
        ac[t]=ac[t]/count
        test_ac[t]=test(test_iter,net)
    epoch_plot(epoch,L,ac,test_ac)    #画出迭代函数曲线
    print("mnist_train accuracy:",ac[epoch-1])          #输出训练集分类准确率
    print("mnist_test accuracy:",test_ac[epoch-1])      #输出测试集分类准确率
    return net  

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    return image

def test(test_iter,net):
    test_ac=0
    count=0
    for X, y in test_iter:
        y_hat = net(X)
        test_ac+=accuracy(y_hat,y)
        count+=1
    return test_ac/count

if __name__ == '__main__':
    alpha=0.001
    epoch=10
    net=LeNet()
    train_iter,test_iter=read_data()
    net=train_LeNet(train_iter,test_iter,net,epoch,alpha)
