import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import svm   #用来画margin边缘
def generate_data(number):                                     #生成数据并将其存到csv文件中
    if number==1:                                              #如果number=1，则生成线性可分的数据集
        mean1=[-5,0]
        mean2=[0,5] 
    if number==2:                                              #如果number=2，则生成线性不可分的数据集
        mean1=[3,0]
        mean2=[0,3]
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

def data_generator():
    X1 = np.array([[119.28, 26.08],     # 福州
                   [121.47, 31.23],     # 上海
                   [118.06, 24.27],     # 厦门
                   [122.10, 37.50],     # 威海
                   [121.31, 25.03],     # 台北
                   [121.46, 39.04],     # 大连
                   [124.23, 40.07],     # 丹东
                   [118.22,31.14],      # 南京    
                   [113.41,29.58],      # 武汉
                   [112.59,28.12],      # 长沙
                   [115.27,28.09]])     # 南昌

    X2 = np.array([[129.87, 32.75],     # 长崎
                   [130.24, 33.35],     # 福冈
                   [130.33, 31.36],     # 鹿儿岛
                   [131.42, 31.91],     # 宫崎
                   [133.33, 15.43],     # 鸟取
                   [138.38, 34.98],     # 静冈
                   [140.47, 36.37]])    # 水户
    X1=np.insert(X1,0,1,axis=1)                            #增广后的数据集x1
    X2=np.insert(X2,0,-1,axis=1)                            #增广后的数据集x1
    city = np.array(['福州', '上海', '厦门', '威海', '台北', '大连', '丹东','南京','武汉','长沙','南昌',
                     '长崎', '福冈', '鹿儿岛', '宫崎', '鸟取', '静冈', '水户'])

    y1 = np.full(X1.shape[0], 1)    # 中国
    y2 = np.full(X2.shape[0], -1)   # 日本
    X_test = np.array([[123.28, 25.45]])    # 钓鱼岛
    y_test = np.array([1])
    X_train = np.concatenate((X1, X2), axis=0)
    y_train = np.concatenate((y1, y2))  # 组合
    X_train, y_train, city = shuffle(X_train, y_train, city)    # 随机排列

    return X_train, y_train, city, X_test, y_test

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

def draw_high(data,label):
    clf = svm.SVC(kernel='linear')
    clf.fit(data, label)
    w = clf.coef_[0]
    a = -w[1] / w[2]    
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[2]
    print(w)
    print(clf.intercept_[0])
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors   
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[2] - a * b[1])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[2] - a * b[1])
    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.scatter(data[:, 1], data[:, 2], c=label, cmap=plt.cm.Paired)
    plt.scatter(clf.support_vectors_[:, 1], clf.support_vectors_[:, 2],c='r')
    plt.axis('tight')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("SVM")
    plt.show()

def diaoyu(X, y, city, X_test, y_test):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    p1 = 0
    p2 = 0
    p3 = 0
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    w = clf.coef_[0]
    a = -w[1] / w[2]    
    xx = np.linspace(115, 145)
    yy = a * xx - (clf.intercept_[0]) / w[2]

    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[2] - a * b[1])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[2] - a * b[1])
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    
    for i in range(len(y)):
        if y[i] == 1:
            p1 = plt.scatter(X[i, 1], X[i, 2], c='r', marker='o', s = 10)    #类别1的点集
            plt.text(X[i, 1], X[i, 2], city[i], fontsize=9, color="r", style="italic", weight="light", verticalalignment='center',
                    horizontalalignment='left')
        else:
            p2 = plt.scatter(X[i, 1], X[i, 2], c='b', marker='x', s = 10)    #类别2的点集
            plt.text(X[i, 1], X[i, 2], city[i], fontsize=9, color="b", style="italic", weight="light",verticalalignment='center',
                    horizontalalignment='left')

    for i in range(len(y_test)):
        if y_test[i] == 1:
            p3 = plt.scatter(X_test[i, 0], X_test[i, 1], c='r', s = 100, marker='*')  # 类别1的点集
        else:
            p3 = plt.scatter(X_test[i, 0], X_test[i, 1], c='b', s = 100,marker='*')  # 类别2的点集

    plt.legend([p1, p2, p3], ['中国', '日本', '钓鱼岛'], loc = 'upper right')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("SVM")
    plt.ylim(15, 45)
    plt.show()
    

