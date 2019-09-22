import pandas as pd 
import numpy as np 

##########################################################################################################

#Fisher算法
def fisher(X1,X2,n,f): #n为样本维数,f为留一法取出的测试样本所在类别
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
		for i in range(0,96):
			S1+=np.dot((X1[i].T-m1),((X1[i].T-m1).T))
		for i in range(0,111):
			S2+=np.dot((X2[i].T-m2),((X2[i].T-m2).T)) 
	if f==1: #取出的测试样本属于X2
		for i in range(0,97):
			S1+=np.dot((X1[i].T-m1),((X1[i].T-m1).T))
		for i in range(0,110):
			S2+=np.dot((X2[i].T-m2),((X2[i].T-m2).T))
	#求Sw
	Sw=S1+S2
	#求最佳投影方向W
	W=np.dot(np.linalg .inv(Sw),(m1-m2))
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

#导入sonar数据集
sonar=pd.read_csv('sonar.all-data',header=None,sep=',')
s=sonar.iloc[0:208,0:60]
D=np.mat(s) #生成数据集矩阵

#留一法划分训练集、测试集(n为维数)
Accuracy=[0]*60 #存放正确率值
for n in range(1,61):
	D1=D[0:97,0:n]
	D2=D[97:208,0:n]

	flag=0 #flag计数正确分类的样本数
	x1=[0]*97
	x2=[0]*111 #存放一维化数据
	for i in range(0,208):
		if i<97: #测试样本在D1类样本中
			text=D1[i] 
			train=np.delete(D1,i,axis=0)
			W,W0=fisher(train,D2,n,0)
			if classify(text,W,W0)>=0: #text样本分在D1类中
				flag+=1 #分类正确
			x1[i]=classify(text,W,W0) #第1类样本的分类投影结果
		else: #测试样本在D2类样本中
			text=D2[i-97]
			train=np.delete(D2,i-97,axis=0)
			W,W0=fisher(D1,train,n,1)
			if classify(text,W,W0)<0: #text样本分在D2类中
				flag+=1 #分类正确
			x2[i-97]=classify(text,W,W0) #第2类样本的分类投影结果
	#计算分类正确率
	accuracy=flag/208 
	Accuracy[n-1]=accuracy
	#打印结果
	print("数据为%d维时，分类正确率为:%.4f"%(n,accuracy))

##########################################################################################################

#数据可视化
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(2,1,1)

#画图：对60维进行投影的结果：
ax.set(xlim=[-0.1,0.1],ylim=[-0.4,0.4],title='60D Classification Results',ylabel='Y',xlabel='sonar')
y1=np.zeros(97)
y2=np.zeros(111)
plt.scatter(x1,y1,color='red',marker='+')
plt.scatter(x2,y2,color='blue',marker='+')

#画图：准确率变化情况：
ax1=fig.add_subplot(2,1,2)
ax1.set(xlim=[0,60],ylim=[0.5,0.8],title='Accuracy variation',ylabel='Accuracy',xlabel='Dimension')
x=np.arange(1,61,1)
plt.plot(x,Accuracy,color='red')

#画图
plt.show()




