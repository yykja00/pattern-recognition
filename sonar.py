#代码因临时更改作业要求，在原有基础上加入了KNN的相关变量，且开始的Fisher变量难以用到KNN中，故较为看起来较为复杂
#KNN相关变量data_all，y_all，X_train，y_train

import xlrd
import numpy as np
import random
from sklearn.preprocessing import  StandardScaler
from sklearn.neighbors import KNeighborsClassifier

wb = xlrd.open_workbook('E:\\sonar.xlsx')
sheet = wb.sheet_by_name("Sheet1")

data_all = np.zeros((60,208))   #纵向读取，避开字母行
for i in range (60):
    data_all[i] = sheet.col_values(i)
data_all = data_all.T   #转置data_all，恢复成正常形态，1-97为R，98-208为M

X1_all = np.zeros((97,60))
X2_all = np.zeros((111,60))
y_all = np.zeros(208)
for i in range(97):
    y_all[i] = np.array(1)
for i in range(111):
    y_all[i+97] = np.array(2)

for i in range(97):         #获得分类好的第一类和第二类
    X1_all[i] = data_all[i]
for i in range(111):
    X2_all[i] = data_all[i+97]

def fisher(X1,X2,n):  #两类数据，维度数
    m1 = (np.mean(X1, axis=0))  #两类数据的均值向量
    m2 = (np.mean(X2, axis=0))
    m1 = m1.reshape(n, 1)  # 将行向量转换为列向量
    m2 = m2.reshape(n, 1)

    S1 = np.zeros((n, n))  #新建两个离散度矩阵
    S2 = np.zeros((n, n))

    for i in range(0,int(97*0.4-1)):  #样本总数的40%
        S1 += (X1[i].reshape(n, 1) - m1).dot((X1[i].reshape(n, 1) - m1).T)
    for i in range(0,int(111*0.4-1)):
        S2 += (X2[i].reshape(n, 1) - m2).dot((X2[i].reshape(n, 1) - m2).T)

    Sw = S1+S2
    W = np.linalg.inv(Sw).dot(m1 - m2)
    m_1 = (W.T).dot(m1) #一维的投影均值
    m_2 = (W.T).dot(m2)
    W0 = 0.5*(m_1 + m_2)

    return W,W0

def classify(X,W,W0):   #判别函数
    y = (W.T).dot(X) - W0
    return y

if __name__ == '__main__':
    oa = 0
    pe_a=0
    aa1_a=0
    aa2_a=0
    for i in range(10):#10次下的训练
        wrong1=0
        wrong2=0
        X1 = np.zeros((37,60))       #建立训练矩阵
        X2 = np.zeros((43,60))
        y_train = np.zeros(80)
        for i in range(37):
            y_train[i]=np.array(1)
        for i in range(43):
            y_train[i+37] = np.array(2)
        X_train = np.append(X1,X2,axis=0)

        a = random.sample(range(0,96),37)   #第一类数据中取得20个进行训练
        for i in range(37):                 #注：使用randint生成的数会有重复的，而使用sample才会生成不重复的随机数
            X1[i] = X1_all[a[i]]
        b = random.sample(range(0,110),43)
        for i in range(int(43)):
            X2[i] = X2_all[b[i]]

        W,W0=fisher(X1,X2,60)
        right = 0
        con_1=0
        for i in range(97):
            if (i not in a):
                if(classify(X1_all[i],W,W0)>0):
                    right=right+1
                    con_1=con_1+1
                else:
                    wrong2=wrong2+1
        for i in range(111):
            if (i not in b):
                if(classify((X2_all[i]),W,W0)<0):
                    right=right+1
                else:
                    wrong1=wrong1+1
        oa = right/128+oa
        pe=(60*(con_1+wrong1)+68*(right-con_1+wrong2))/(128*128)
        aa1=con_1/60
        aa2=(right-con_1)/68
        aa1_a=aa1+aa1_a
        aa2_a=aa2+aa2_a
        pe_a=pe_a+pe
    #循环结束
    OA=oa/10
    AA1=aa1_a/10
    AA2=aa2_a/10
    PE=pe_a/10
    K=(OA-PE)/(1-PE)
    print("Fisher判决的OA的值为:%.3f"%(OA))
    print("Fisher判决的各类的AA值分别为:%.3f,%.3f"%(AA1,AA2))
    print("Fisher判决的Kappa系数为:%.3f"%(K))

    # KNN分类
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(data_all)
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    y_predict = knc.predict(data_all)
    print('KNN分类的正确率为：%.3f'%(knc.score(X_test, y_all)))