#代码因临时更改作业要求，在原有基础上加入了KNN的相关变量，且开始的Fisher变量难以用到KNN中，故较为看起来较为复杂
#KNN相关变量X_all，y_all，X_train，y_train


import xlrd
import numpy as np
import random
from sklearn.preprocessing import  StandardScaler
from sklearn.neighbors import KNeighborsClassifier


wb = xlrd.open_workbook('E:\\iris.xlsx')
sheet = wb.sheet_by_name("Sheet1")



X1_all = np.zeros((50,5))        #将数据存入3个矩阵中
X2_all = np.zeros((50,5))
X3_all = np.zeros((50,5))

for i in range (50):
    X1_all[i] = sheet.row_values(i)
for i in range (50):
    X2_all[i] = sheet.row_values(i+50)
for i in range (50):
    X3_all[i] = sheet.row_values(i+100)
X1_all = np.delete(X1_all,-1,axis=1)
X2_all = np.delete(X2_all,-1,axis=1)
X3_all = np.delete(X3_all,-1,axis=1)
X_all = np.append(X1_all,X2_all,axis=0)
X_all = np.append(X_all,X3_all,axis=0)
y_all = np.zeros(150)
for i in range(50):
    y_all[i] = np.array(0)
for i in range(50):
    y_all[i+50] = np.array(1)
for i in range(50):
    y_all[i+100] = np.array(2)
y_train = np.zeros(60)
for i in range(20):
    y_train[i] = np.array(0)
for i in range(20):
    y_train[i+20] = np.array(1)
for i in range(20):
    y_train[i+40] = np.array(2)

def fisher(X1,X2,n):  #两类数据，维度数
    m1 = (np.mean(X1, axis=0))  #两类数据的均值向量
    m2 = (np.mean(X2, axis=0))
    m1 = m1.reshape(n, 1)  # 将行向量转换为列向量
    m2 = m2.reshape(n, 1)

    S1 = np.zeros((n, n))  #新建两个离散度矩阵
    S2 = np.zeros((n, n))

    for i in range(0,19):  #只有20个训练样本
        S1 += (X1[i].reshape(n, 1) - m1).dot((X1[i].reshape(n, 1) - m1).T)
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
    aa1_a=0
    aa2_a=0
    aa3_a=0
    pe_all=0
    for i in range(10):#10次下的训练
        wrongA=0
        wrongB=0
        wrongC=0
        X1 = np.zeros((20,4))        #建立训练矩阵
        X2 = np.zeros((20,4))
        X3 = np.zeros((20,4))
        a = random.sample(range(0,49),20)   #第一类数据中取得20个进行训练
        for i in range(20):                 #注：使用randint生成的数会有重复的，而使用sample才会生成不重复的随机数
            X1[i] = X1_all[a[i]]
        b = random.sample(range(0,49),20)
        for i in range(20):
            X2[i] = X2_all[b[i]]
        c = random.sample(range(0,49),20)
        for i in range(20):
            X3[i] = X3_all[c[i]]
        W1,W01 = fisher(X1,X2,4)#各类分类决策
        W2,W02 = fisher(X2,X3,4)
        W3,W03 = fisher(X3,X1,4)
        X_train = np.append(X1,X2,axis=0)
        X_train = np.append(X_train,X3,axis=0)

        right=0
        for i in range(50):
            if (i not in a):

                A=0
                B=0
                C=0
                if(classify(X1_all[i],W1,W01)>0):
                    A=A+1
                else:
                    B=B+1
                if(classify(X1_all[i],W2,W02)>0):
                    B=B+1
                else:
                    C=C+1
                if(classify(X1_all[i],W3,W03)>0):
                    C=C+1
                else:
                    A=A+1
                if(A>=B and A>=C):
                    right=right+1
                aa1 = right/30
                if(B>A and B>=C):
                    wrongB=wrongB+1
                if(C>A and C>B):
                    wrongC=wrongC+1

        for i in range(50):
            if (i not in b):
                A = 0
                B = 0
                C = 0
                if (classify(X2_all[i], W1, W01) > 0):
                    A =A+ 1
                else:
                    B =B+ 1
                if (classify(X2_all[i], W2, W02) > 0):
                    B =B+ 1
                else:
                    C =C+ 1
                if (classify(X2_all[i], W3, W03) > 0):
                    C =C+ 1
                else:
                    A =A+ 1
                if (B >= A and B >= C):
                    right =right+ 1
                if(A>B and A>=C):
                    wrongA=wrongA+1
                if(C>A and C>B):
                    wrongC=wrongC+1
                aa2 = (right-aa1*30)/30
        for i in range(50):
            if (i not in c):
                A = 0
                B = 0
                C = 0
                if (classify(X3_all[i], W1, W01) > 0):
                    A=A + 1
                else:
                    B =B+ 1
                if (classify(X3_all[i], W2, W02) > 0):
                    B =B+ 1
                else:
                    C =C+ 1
                if (classify(X3_all[i], W3, W03) > 0):
                    C =C+ 1
                else:
                    A =A+ 1
                if (C >= B and C >= A):
                    right=right + 1
                if(A>=B and A>C):
                    wrongA=wrongA+1
                if(B>C and B>A):
                    wrongB=wrongB+1
                aa3 = (right-aa1*30-aa2*30)/30
        pe=(30*(aa1+wrongA)+30*(aa2+wrongB)+30*(aa3+wrongC))/(90*90)
        oa=right/90+oa
        aa1_a=aa1+aa1_a
        aa2_a=aa2+aa2_a
        aa3_a=aa3+aa3_a
        pe_all=pe_all+pe
    #循环结束
    OA=oa/10
    AA1=aa1_a/10
    AA2=aa2_a/10
    AA3=aa3_a/10
    PE=pe_all/10
    K=(OA-PE)/(1-PE)
    print("Fisher函数判决的OA值为:%.3f"%(OA))
    print("Fisher函数判决的三类的AA值分别为:%.3f,%.3f,%.3f"%(AA1,AA2,AA3))
    print("Fisher函数判决的Kappa系数为:%.3f"%(K))

    # KNN分类
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_all)
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_all)
    print('KNN分类的正确率为：%.3f'%(knc.score(X_test,y_all)))