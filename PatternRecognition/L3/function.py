import numpy as np
import math 
import matplotlib.pyplot as plt
def function(x):
    return x*math.cos(0.25*math.pi*x)

def gradient(x):
    return math.cos(0.25*math.pi*x)-0.25*math.pi*x*math.sin(0.25*math.pi*x)

def draw(X,f):
    t=np.linspace(-4.5,5)
    y=np.ones(len(t))
    for i in range(len(t)):
        y[i]=function(t[i])
    for i in range(len(f)):
        plt.scatter(X[i],f[i],c="r")
    plt.title('Variation of Adam x and f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.plot(t,y)
    plt.plot(X,f)
    plt.show()

def Gradient_Descent(x,epoch,alpha):
    f=np.zeros(epoch+1)
    X=np.zeros(epoch+1)
    f[0]=function(x)
    X[0]=x
    for t in range(epoch):
        x-=alpha*gradient(x)
        X[t+1]=x
        f[t+1]=function(x)
    draw(X,f)
    
def Adagrad(x,epoch,alpha):
    f=np.zeros(epoch+1)
    X=np.zeros(epoch+1)  
    w=np.zeros(epoch)
    f[0]=function(x)
    X[0]=x
    for t in range(epoch):
        w[t]=gradient(x)
        zeta=0
        for k in range(epoch):
            zeta+=w[k]**2/(t+1)
        zeta=math.sqrt(zeta+1e-6)
        x-=alpha*gradient(x)/zeta
        X[t+1]=x
        f[t+1]=function(x)
    draw(X,f)

def RMSProp(x,epoch,alpha,k):
    f=np.zeros(epoch+1)
    X=np.zeros(epoch+1)  
    zeta=np.zeros(epoch)
    f[0]=function(x)
    X[0]=x
    for t in range(epoch):
        zeta[t]=math.sqrt(k*zeta[t-1]**2+(1-k)*gradient(x)**2)
        x-=alpha*gradient(x)/zeta[t]
        X[t+1]=x
        f[t+1]=function(x)
    draw(X,f)

def Moment(x,epoch,alpha,lamda):
    f=np.zeros(epoch+1)
    X=np.zeros(epoch+1) 
    moment=np.zeros(epoch) 
    f[0]=function(x)
    X[0]=x
    for t in range(epoch):
        moment[t]=lamda*moment[t-1]-alpha*gradient(x)
        x+=moment[t]
        X[t+1]=x
        f[t+1]=function(x)
    draw(X,f)
    
def Adam(x,epoch,alpha,beta1,beta2):
    f=np.zeros(epoch+1)
    X=np.zeros(epoch+1) 
    moment=np.zeros(epoch) 
    v=np.zeros(epoch)
    f[0]=function(x)
    X[0]=x
    for t in range(epoch):
        moment[t]=beta1*moment[t-1]+(1-beta1)*gradient(x)
        v[t]=beta2*v[t-1]+(1-beta2)*gradient(x)**2
        mtt=moment[t]/(1-beta1**(t+1))
        vtt=v[t]/(1-beta2**(t+1))
        x-=alpha*mtt/(math.sqrt(vtt)+1e-6)
        X[t+1]=x
        f[t+1]=function(x)
    draw(X,f)

if __name__ == '__main__':
    x=-4
    alpha=0.4
    k=0.9
    lamda=0.9
    epoch=10
    beta1=0.9
    beta2=0.999
    #Gradient_Descent(x,epoch,alpha)
    #Adagrad(x,epoch,alpha)
    #RMSProp(x,epoch,alpha,k)
    #Moment(x,epoch,alpha,lamda)
    Adam(x,epoch,alpha,beta1,beta2)