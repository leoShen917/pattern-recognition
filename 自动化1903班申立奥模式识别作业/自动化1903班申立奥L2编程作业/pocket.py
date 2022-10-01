import numpy as np
import math
import time
import data_process as dp
class Pocket:
    def __init__(self,datadim):                           
        self.w=np.zeros(datadim)                          #初始化权重w=0  增广化
        
    def check(self,data,label,w):
        s=np.dot(data,w)
        y_pred=np.ones_like(label)
        loc_n=np.where(s<0)[0]
        y_pred[loc_n]=-1
        wrong=np.where(label!=y_pred)[0]
        fault=len(wrong)
        return fault,wrong

    def train(self,data,label,num): 
        w=np.zeros(len(data[0]))
        for i in range(num):
            fault,wrong=self.check(data,label,w)
            if fault==0:
                break
            else:
                pos=np.random.choice(wrong)
                wt=w+label[pos]*data[pos]
                new_fault,new_wrong=self.check(data,label,wt)
                if new_fault<fault:
                    w=wt
        self.w=w
    
    def test(self,all_data,all_label,testdata,testlabel): #分类器在测试集中的准确率
        count=0
        length=len(testdata)
        y=np.zeros(length)
        dp.draw(all_data,all_label,self.w,1)              #画出分类面和数据集
        for i in range(length):
            y[i] = np.sign(np.dot(self.w.T,testdata[i]))
            if y[i]==testlabel[i]:
                count+=1
        precision=count/length
        return precision

if __name__ == '__main__':
    num=2                                                 #不同的num对应读取不同的数据集
    all_data,all_label=dp.read_data(num)                  #提取数据集
    train_data,train_label=dp.split_train(num)            #训练
    test_data,test_label=dp.split_test(num)               #测试集
    datadim=len(all_data[0])                
    start=time.perf_counter()                             #计时
    pocket=Pocket(datadim)
    pocket.train(train_data,train_label,1000)                         #训练模型
    precision=pocket.test(all_data,all_label,test_data,test_label)   #预测
    end=time.perf_counter()
    dur = end - start
    print(precision)
    print('Running time: %s Seconds'%(dur))
    






