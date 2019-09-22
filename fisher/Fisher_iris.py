2import pandas as pd 
import numpy as np 

##########################################################################################################

#Fisher算法
def fisher(X1,X2,n,f):#n为样本维数,f为留一法取出的测试样本所在类别
	"""
	求X1、X2均值向量m1、m2
	样本类内离散度矩阵S1、S2
	求Sw
	求W、W0
	"""
	m1=np.mean(X1,axis=0)
	m2=np.mean(X2,axis=0)
	m1=m1.T
	m2=m2.T #转换成列向量

	S1=np.zeros((n,n)) 
	S2=np.zeros((n,n))
	if f==0: #取出的测试样本属于X1
		for i in range(0,48):
			S1+=np.dot((X1[i].T-m1),((X1[i].T-m1).T))
		for i in range(0,49):
			S2+=np.dot((X2[i].T-m2),((X2[i].T-m2).T)) 
	if f==1: #取出的测试样本属于X2
		for i in range(0,49):
			S1+=np.dot((X1[i].T-m1),((X1[i].T-m1).T))
		for i in range(0,48):
			S2+=np.dot((X2[i].T-m2),((X2[i].T-m2).T))
	#求Sw
	Sw=S1+S2
	#求最佳投影方向W
	W=np.dot(np.linalg.inv(Sw),(m1-m2))
	#计算分类阈值W0
	m_1=np.dot((W.T),m1)
	m_2=np.dot((W.T),m2)
	W0=-0.5*(m_1+m_2)
	return W,W0

def classify(X,W,W0):
	"""
	将X分类，结果返回判别函数值
	"""
	X=X.T
	g_x=np.dot((W.T),X)+W0 #求判别函数g_x
	return g_x

##########################################################################################################

#导入iris数据集
iris=pd.read_csv('iris.data',header=None,sep=',')
i=iris.iloc[0:150,0:4]
D=np.mat(i) #生成数据集矩阵

#分类第1、2类
#留一法划分训练集、测试集(n为维数)
Accuracy12=[0]*4 #存放1、2类分类正确率值
for n in range(1,5):
	D1=D[0:50,0:n]
	D2=D[50:100,0:n]

	flag=0 #flag计数正确分类的样本数
	x12_1=[0]*50
	x12_2=[0]*50 #存放1、2类一维化数据
	for i in range(0,100):
		if i<50: #测试样本在D1类样本中
			text=D1[i] 
			train=np.delete(D1,i,axis=0)
			W,W0=fisher(train,D2,n,0)
			if classify(text,W,W0)>=0: #text样本分在D1类中
				flag+=1 #分类正确
			x12_1[i]=classify(text,W,W0)
		else: #测试样本在D2类样本中
			text=D2[i-50]
			train=np.delete(D2,i-50,axis=0)
			W,W0=fisher(D1,train,n,1)
			if classify(text,W,W0)<0: #text样本分在D2类中
				flag+=1 #分类正确
			x12_2[i-50]=classify(text,W,W0)
	#计算分类正确率
	accuracy=flag/100 
	Accuracy12[n-1]=accuracy
	#打印结果
	print("分类第1、2类：数据为%d维时，分类正确率为：%.4f"%(n,accuracy))


#分类第2、3类
#留一法划分训练集、测试集(n为维数)
Accuracy23=[0]*4 #存放2、3类分类正确率值
for n in range(1,5):
	D1=D[50:100,0:n]
	D2=D[100:150,0:n]

	flag=0 #flag计数正确分类的样本数
	x23_1=[0]*50
	x23_2=[0]*50 #存放2、3类一维化数据
	for i in range(0,100):
		if i<50: #测试样本在D1类样本中
			text=D1[i] 
			train=np.delete(D1,i,axis=0)
			W,W0=fisher(train,D2,n,0)
			if classify(text,W,W0)>=0: #text样本分在D1类中
				flag+=1 #分类正确
			x23_1[i]=classify(text,W,W0) #第2类样本的分类投影结果
		else: #测试样本在D2类样本中
			text=D2[i-50]
			train=np.delete(D2,i-50,axis=0)
			W,W0=fisher(D1,train,n,1)
			if classify(text,W,W0)<0: #text样本分在D2类中
				flag+=1 #分类正确
			x23_2[i-50]=classify(text,W,W0) #第3类样本的分类投影结果
	#计算分类正确率
	accuracy=flag/100 
	Accuracy23[n-1]=accuracy
	#打印结果
	print("分类第2、3类：数据为%d维时，分类正确率为：%.4f"%(n,accuracy))

#分类第1、3类
#留一法划分训练集、测试集(n为维数)
Accuracy13=[0]*4 #存放1、3类分类正确率值
for n in range(1,5):
	D1=D[0:50,0:n]
	D2=D[100:150,0:n]

	flag=0 #flag计数正确分类的样本数
	x13_1=[0]*50
	x13_2=[0]*50 #存放1、3类一维化数据
	for i in range(0,100):
		if i<50: #测试样本在D1类样本中
			text=D1[i] 
			train=np.delete(D1,i,axis=0)
			W,W0=fisher(train,D2,n,0)
			if classify(text,W,W0)>=0: #text样本分在D1类中
				flag+=1 #分类正确
			x13_1[i]=classify(text,W,W0) #第1类样本的分类投影结果
		else: #测试样本在D2类样本中
			text=D2[i-50]
			train=np.delete(D2,i-50,axis=0)
			W,W0=fisher(D1,train,n,1)
			if classify(text,W,W0)<0: #text样本分在D2类中
				flag+=1 #分类正确
			x13_2[i-50]=classify(text,W,W0) #第3类样本的分类投影结果
	#计算分类正确率
	accuracy=flag/100 
	Accuracy13[n-1]=accuracy
	#打印结果
	print("分类第1、3类：数据为%d维时，分类正确率为：%.4f"%(n,accuracy))

##########################################################################################################

#数据可视化
import matplotlib.pyplot as plt
fig=plt.figure()

#画图：对4维进行投影的结果：
#1、2类
ax12=fig.add_subplot(2,3,1) #确定位置
ax12.set(xlim=[-1,1],ylim=[-0.4,0.4],title='4D Classification 12 Results',ylabel='Y',xlabel='iris') 
y_1=np.zeros(50)
y_2=np.zeros(50)
plt.scatter(x12_1,y_1,color='red',marker='+')
plt.scatter(x12_2,y_2,color='blue',marker='+')

#2、3类
ax23=fig.add_subplot(2,3,2)
ax23.set(xlim=[-0.5,0.5],ylim=[-0.4,0.4],title='4D Classification 23 Results',ylabel='Y',xlabel='iris')
plt.scatter(x23_1,y_1,color='red',marker='+')
plt.scatter(x23_2,y_2,color='blue',marker='+')

#1、3类
ax13=fig.add_subplot(2,3,3)
ax13.set(xlim=[-2,2],ylim=[-0.4,0.4],title='4D Classification 13 Results',ylabel='Y',xlabel='iris')
plt.scatter(x13_1,y_1,color='red',marker='+')
plt.scatter(x13_2,y_2,color='blue',marker='+')

#画图：准确率变化情况：
#1、2类
ax1=fig.add_subplot(2,3,4)
ax1.set(xlim=[0,4],ylim=[0.7,1.0],title='Accuracy12 variation',ylabel='Accuracy12',xlabel='Dimension')
x=np.arange(1,5,1)
plt.plot(x,Accuracy12,color='red')

#2、3类
ax2=fig.add_subplot(2,3,5)
ax2.set(xlim=[0,4],ylim=[0.7,1.0],title='Accuracy23 variation',ylabel='Accuracy23',xlabel='Dimension')
plt.plot(x,Accuracy23,color='red')

#1、3类
ax3=fig.add_subplot(2,3,6)
ax3.set(xlim=[0,4],ylim=[0.7,1.0],title='Accuracy13 variation',ylabel='Accuracy13',xlabel='Dimension')
plt.plot(x,Accuracy13,color='red')

#画图
plt.show()