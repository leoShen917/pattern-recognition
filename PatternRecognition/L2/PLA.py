from re import L
import numpy as np
import time
import data_process as dp

class PLA:
    def __init__(self,datadim):  #初始化参数
        self.datadim=datadim  #输入样本维数
        self.w=np.zeros(self.datadim) #初始化权重w=0  增广化

    def train(self,data,label):#训练样本集，寻找分类面
        #dp.draw(train_data,train_label,self.w,0)
        length=len(data)
        y=np.zeros(length)
        i=0
        while i<length:
            y[i]=np.sign(np.dot(self.w,data[i].T))
            if y[i]!=label[i]:
                self.w = self.w+label[i]*data[i].T
                #dp.draw(train_data,train_label,self.w,0)
                i=0
            else:
                i+=1  
        '''        
        count=0
        for i in range(length):
            y[i] = np.sign(np.dot(self.w.T,data[i]))
            if y[i]==label[i]:
                count+=1
        precision=count/length
        '''
        return self.w

    def predict(self,all_data,all_label,testdata,testlabel): #分类器在测试集中的准确率
        count=0
        length=len(testdata)
        y=np.zeros(length)
        dp.draw(all_data,all_label,self.w,1)                 #画出分类面和数据集
        for i in range(length):
            y[i] = np.sign(np.dot(self.w,testdata[i].T))
            if y[i]==testlabel[i]:
                count+=1
        precision=count/length
        return precision

if __name__ == '__main__':
    num=1                                                     #不同的num对应读取不同的数据集
    all_data,all_label=dp.read_data(num)                      #提取数据集
    train_data,train_label=dp.split_train(num)                #训练集
    test_data,test_label=dp.split_test(num)                   #测试集
    datadim=len(all_data[0])
    start=time.perf_counter()                                 #计时
    pla=PLA(datadim)
    pla.train(train_data,train_label)                               #训练模型
    precision=pla.predict(all_data,all_label,test_data,test_label)  #预测
    end=time.perf_counter()
    dur = end - start
    print(precision)
    print('Running time: %s Seconds'%(dur))

