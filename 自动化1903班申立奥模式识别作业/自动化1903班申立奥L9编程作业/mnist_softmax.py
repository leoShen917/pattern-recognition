import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def softmax(X):                                 #进行softmax处理
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

def cross_entropy(y_hat, y):                    #计算交叉熵
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))

def read_data():                                #读取mnist文件
    mnist_train=torchvision.datasets.MNIST(root="./dataset",train=True,download=True,transform=transforms.ToTensor())
    mnist_test=torchvision.datasets.MNIST(root="./dataset",train=False,download=True,transform=transforms.ToTensor())
    batch_size = 256
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return train_iter,test_iter

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

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    return image

class mnist_softmax:
    def __init__(self,train_iter,test_iter):    #初始化参数
        self.num_inputs = 784
        self.num_outputs = 10
        self.w = torch.tensor(np.random.normal(0, 0.01, (self.num_inputs, self.num_outputs)), dtype=torch.float)
        self.b = torch.zeros(self.num_outputs, dtype=torch.float)
        self.train_iter=train_iter
        self.test_iter=test_iter

    def Model(self,X):                          #softmax模型
        return softmax(torch.mm(X.view((-1, self.num_inputs)), self.w) + self.b)

    def accuracy(self,y_hat, y):                #计算分类正确率
        return (y_hat.argmax(dim=1) == y).float().mean().item()

    def train(self,epoch,alpha):
        t=0
        
        ac=np.zeros(epoch)
        L=np.zeros(epoch)
        test_ac=np.zeros(epoch)
        while t<epoch:
            count=0
            for X, y in self.train_iter:    #读取每个batch中的数据
                cnt=0
                y_hat = self.Model(X)
                L[t]+=cross_entropy(y_hat, y).sum()
                ac[t]+=self.accuracy(y_hat,y)
                one_hot = torch.zeros(len(y), 10)
                #转换为独热编码
                for i in one_hot:
                    i[y[cnt]] = 1
                    cnt += 1
                #进行权系数更新
                b_diff=1/X.shape[0]*torch.mm(torch.ones(1,X.shape[0]),(y_hat-one_hot))
                w_diff=1/X.shape[0]*torch.mm(X.view((-1, self.num_inputs)).T,(y_hat-one_hot))
                w_t=self.w-alpha*w_diff
                b_t=self.b-alpha*b_diff
                
                deltaw=torch.norm(w_t-self.w)
                deltab=torch.norm(b_t-self.b)
                
                #计算错分样本数
                misindex = torch.nonzero(torch.argmax(y_hat, axis = 1)!=y)
                misnum = len(misindex)
                #如果错分数或者梯度为0，则停止迭代
                if misnum==0 or (deltaw+deltab<1e-5):
                    break
                else:
                    self.w = w_t          # 否则, 继续迭代
                    self.b = b_t
                count+=1
                
            L[t]=L[t]/count
            ac[t]=ac[t]/count
            test_ac[t]=self.test()
            t+=1   
        epoch_plot(epoch,L,ac,test_ac)    #画出迭代函数曲线
        print("mnist_train accuracy:",ac[epoch-1])
        print("mnist_test accuracy:",test_ac[epoch-1])
        
        X, y = iter(test_iter).next()
        label=torch.zeros(10)
        hat=torch.zeros(10)
        label_hat=self.Model(X)
        for i in range(10):
            img=tensor_to_PIL(X[i])
            img.show()
            label[i]=y[i]
            hat[i]=torch.argmax(label_hat[i]).item()
            img.save("%d.png"%i)
        print(label,hat)

    def test(self):
        test_ac=0
        count=0
        for X, y in self.test_iter:
            y_hat = self.Model(X)
            test_ac+=self.accuracy(y_hat,y)
            count+=1
        return test_ac/count



if __name__ == '__main__':
    train_iter,test_iter=read_data()
    print(train_iter)
    mnist=mnist_softmax(train_iter,test_iter)
    mnist.train(10,1)  #参数为迭代次数和学习率
    