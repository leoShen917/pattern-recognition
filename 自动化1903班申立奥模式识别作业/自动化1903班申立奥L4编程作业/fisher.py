import numpy as np
import data_process as dp
def fishertest(train_data,train_label):
    datadim=len(train_data[0])                                      #数据维数datadim
    data1=train_data[np.where(train_label==1)]                      #data1为“1”类数据
    data2=train_data[np.where(train_label==-1)]                     #data2为“-1”类数据
    u1 = np.mean(data1, axis=0)                                     #数据的均值
    u2 = np.mean(data2, axis=0)                                     
    cov1=np.zeros((datadim,datadim))                                
    cov2=np.zeros((datadim,datadim))
    for i in range(len(data1)):                                     #求数据的协方差矩阵
        cov1+=np.dot(np.array([data1[i]-u1]).T,np.array([data1[i]-u1]))
    for i in range(len(data2)):
        cov2+=np.dot(np.array([data2[i]-u2]).T,np.array([data2[i]-u2]))
    Sw=cov1+cov2                                                    #类内总离差阵
    Swinv=np.linalg.pinv(Sw)                                        #求逆
    w=np.dot(Swinv,np.array([(u1-u2)]).T)                           #最佳投影向量
    w=w.T[0]
    s=np.dot(w.T,np.array([(u1+u2)/2]).T)                           #分类阈值
    return w,s

def predict(all_data,all_label,testdata,testlabel,w,s):             #分类器在测试集中的准确率
    count=0
    length=len(testdata)
    y=np.zeros(length)
    s=s[0]
    dp.draw(all_data,all_label,w,s)                                 #画出分类面和数据集
    for i in range(length):
        y[i] = np.sign(np.dot(w,testdata[i].T)-s)
        if y[i]==testlabel[i]:
            count+=1
    precision=count/length
    return precision

if __name__ == '__main__':
    num=1                                                           #不同的num对应读取不同的数据集，num=1为线性可分数据集，num=2为线性不可分数据集
    all_data,all_label=dp.read_data(num)                            #提取数据集
    train_data,train_label=dp.split_train(num)                      #训练集
    test_data,test_label=dp.split_test(num)                         #测试集        
    w,s=fishertest(train_data,train_label)                            #fisher线性判别
    precision=predict(all_data,all_label,test_data,test_label,w,s)    #测试集上预测样本正确率
    print(precision)

