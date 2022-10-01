import data_process as dp
import numpy as np
class ovo():                                    #定义一对一多分类
    def pla(self,data,label,epoch):             #感知器算法，epoch为最大迭代次数
        w=np.zeros(len(data[0]))
        for i in range(epoch):
            s=np.dot(data,w)
            y_pred=np.ones_like(label)
            loc_n=np.where(s<0)[0]
            y_pred[loc_n]=-1
            wrong=np.where(label!=y_pred)[0]
            num_fault=len(wrong)                #分错样本数量
            if num_fault==0:
                break
            else:
                pos=np.random.choice(wrong)     
                w+=label[pos]*data[pos]         #权重更新
        return w

    def ovo(self,data,label,epoch):             #一对一分类
        species=len(np.unique(label))
        num=int(species*(species-1)/2)          #分类空间数量
        w=np.zeros((num,len(data[0])))
        I=np.ones(len(label))
        count=0
        for i in range(species):                #对训练样本集的重分组
            for j in range(i+1,species):
                index1=np.where(label==i)[0].tolist()   
                index2=np.where(label==j)[0].tolist()
                index=index1+index2
                new_data=data[index]
                #重新打标签，i类为”1“，j类为”-1“
                new_label1=I[index1]            
                new_label2=-I[index2]
                new_label=np.append(new_label1,new_label2)
                w[count]=self.pla(new_data,new_label,epoch) #求出各分类空间的分类面
                count+=1
        return w 

    def vote(self,data,label,w):                #利用投票机制确定样本所属类别
        species=len(np.unique(label))           #标签种类数
        y=np.zeros(len(data))
        for n in range(len(data)):
            count=0
            v=np.zeros(species)
            for i in range(species):
                for j in range(i+1,species):
                    #如果s>0，则属于i类
                    if np.sign(np.dot(w[count],data[n]))>0:
                        v[i]+=1
                    #如果s<0,则属于j类
                    else :
                        v[j]+=1
                    count+=1
            y[n]=np.argmax(v)
        print(y)
        return y

    def predict(self,label,y):                #预测测试样本的正确分类率
        count=0
        for i in range(len(label)):
            if y[i]==label[i]:
                count+=1
        accuracy = count/len(label)
        print("ovo accuracy:",accuracy)

class Softmax():
    def softmax(self,X, w):                     #对数据集进行softmax
        S = np.dot(X, w.T)
        c = np.max(S, axis = 1).reshape(-1, 1)  #在原先的基础上减去最大值来代替Softmax函数，防止溢出，并将c拉成一列
        P = np.exp(S) / \
        np.sum(np.exp(S), axis = 1).reshape(-1, 1)
        return P

    def train(self,data,label,alpha,epoch):
        species=len(np.unique(label))
        np.random.seed()
        w_t = np.random.randn(species, data.shape[1])
        y = np.zeros((len(label), species))
        cnt = 0
        for i in y: # y 转为独热编码
            i[label[cnt]] = 1
            cnt += 1
        for i in range(epoch):
            P = self.softmax(data, w_t)
            diff = 1/data.shape[0] *np.dot((P-y).T, data)
            w_t1=w_t-alpha*diff
            delta = np.linalg.norm(w_t - w_t1)
            misindex = np.where(np.argmax(P, axis = 1) != label)[0]
            misnum = len(misindex)
            if misnum == 0:
                # 对于概率矩阵P，其每一列中最大的值对应的索引，如果均和y一一对应，说明预测完全相同，训练结束
                break
            if delta < (1e-4):      # 或者梯度小于0, 训练结束
                break   
            else:
                w_t = w_t1          # 否则, 继续迭代
        return w_t

    def test(self,data, label, w):
        P = self.softmax(data, w)
        y_pred = np.argmax(P, axis=1)
        misindex = np.where(y_pred != label)[0]
        misnum = len(misindex)  # 出现预测与测试集不一致的个数
        accuracy = (data.shape[0] - misnum) / data.shape[0]   # 精确度
        print("softmax accuracy:",accuracy)
        
if __name__ == '__main__':    
    train_data,train_label=dp.split_train("dataset\iris.csv")    
    test_data,test_label=dp.split_test("dataset\iris.csv")

    OVO=ovo()
    w=OVO.ovo(train_data,train_label,1000)
    y=OVO.vote(test_data,test_label,w)
    OVO.predict(test_label,y)

    soft=Softmax()
    w_t=soft.train(train_data,train_label,0.1,1000)  #迭代次数为1000
    soft.test(test_data, test_label, w_t)
