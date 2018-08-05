#encoding=utf-8
from math import log
import operator
import pickle

#读取数据集
def createDateset(filename):
    with open(filename, 'r')as csvfile:
        dataset= [line.strip().split(', ') for line in csvfile.readlines()]     #读取文件中的每一行
        dataset=[[int(i) if i.isdigit() else i for i in row] for row in dataset]    #对于每一行中的每一个元素，将行列式数字化并且去除空白保证匹配的正确完成
        cleanoutdata(dataset)   #清洗数据
        del (dataset[-1])       #去除最后一行的空行
        precondition(dataset)   #预处理数据
        labels=['workclass','education',
               'marital-status','occupation',
                'relationship','race','sex',
                'native-country']
        return dataset,labels

def cleanoutdata(dataset):#数据清洗
    for row in dataset:
        for column in row:
            if column == '?' or column=='':
                dataset.remove(row)
                break
    for row in dataset:
        for column in row:
            if column == '?' or column=='':
                dataset.remove(row)
                break

#计算香农熵/期望信息
def calculateEntropy(dataset):
    datasetlen=len(dataset)
    labelCount={}#标签统计字典，用来统计每个标签的概率
    for vec in dataset:
        currentlabel=vec[-1]
        if currentlabel not in labelCount.keys():
            labelCount[currentlabel]=0
        labelCount[currentlabel]+=1         #计算出现次数
    shannonEntropy=0.0
    for key in labelCount:
        prob=float(labelCount[key])/datasetlen      #计算概率
        shannonEntropy-=prob*log(prob,2)
    return shannonEntropy

#划分数据集
def splitDataset(dataset,feature,value):
    retdataset=[]
    for vec in dataset:#将选定的feature的列从数据集中去除
        if vec[feature]==value:
            reducedataset=vec[:feature]
            reducedataset.extend(vec[feature+1:])
            retdataset.append(reducedataset)
    return retdataset

#选择最好的数据集划分方式
def chooseBestSplit(dataset):
    featurenum=len(dataset[0])-1    #计算feature的个数，由于dataset中是包含有类别的，所以要减去类别
    shannon=calculateEntropy(dataset)#计算整个数据集的香农熵(期望信息)，用来和每个feature的香农熵进行比较
    bestfeature=0                   #最好的划分方式的索引值，因为0也是索引值，所以应该设置为负数
    gain=0.0                        #信息增益=期望信息-熵,gain为最好的信息增益,split_gain为各种划分方式的信息增益
    for feature in range(featurenum):
        featurelist=[feat[feature] for feat in dataset] #对于dataset中每一个feature，创建单独的列表list，其中是不重复的
        unique=set(featurelist)
        entropy=0.0
        for value in unique:
            subdataset=splitDataset(dataset,feature,value)
            prob=len(subdataset)/float(len(dataset))
            entropy+=prob*calculateEntropy(subdataset)
        split_gain=shannon-entropy        #计算对于该种划分方式的信息增益
        if(split_gain > gain):
            gain=split_gain
            bestfeature=feature
    return bestfeature

#返回出现次数最多的类别，避免产生所有特征全部用完无法判断类别的情况
def majoritycnt(classlist):
    classcount={}
    for i in classlist:
        if i not in classcount.keys():classcount[i]=0
        classcount[i]+=1
    sortedclasscount=sorted(dict2list(classcount),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

#dict字典转换为list列表
def dict2list(dic:dict):
    keys=dic.keys()
    values=dic.values()
    lst=[(key,value)for key,value in zip(keys,values)]
    return lst

#创建树
def createTree(dataset,labels):
    classlist=[feature[-1] for feature in dataset]         #产生数据集中的分类列表，保存的是每一行的分类
    if classlist.count(classlist[0])==len(classlist):       #如果分类别表中的所有分类都是一样的，则直接返回当前的分类
        return classlist[0]
    if len(dataset[0]) == 1:                                #如果划分数据集已经到了无法继续划分的程度，即已经使用完了全部的feature，则进行决策
        return majoritycnt(classlist)
    bestfeature=chooseBestSplit(dataset)                    #计算香农熵和信息增益来返回最佳的划分方案，bestfeature保存最佳的划分的feature的索引
    bestfeaturelabel=labels[bestfeature]                    #取出上述的bestfeature的具体值
    Tree={bestfeaturelabel:{}}
    del(labels[bestfeature])                               #删除当前的feature避免下次继续使用到这个feature来划分
    featurevalue=[feature[bestfeature]for feature in dataset]   #对于上述取出的bestfeature取出数据集中属于当前feature的列的所有的值
    uniquevalue=set(featurevalue)                               #去重
    for value in uniquevalue:                                   #对于每一个feature标签的value值，进行递归构造决策树
        sublabels=labels[:]
        Tree[bestfeaturelabel][value]=createTree(splitDataset(dataset,bestfeature,value),sublabels)
    return Tree

def storetree(inputree,filename):
    fw = open(filename, 'wb')
    pickle.dump(inputree, fw)
    fw.close()

def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

#测试算法
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]        #取出树的第一个标签
    secondDict = inputTree[firstStr]            #取出树的第一个标签下的字典
    featIndex = featLabels.index(firstStr)
    classLabe='<=50K'
    for key in secondDict.keys():#对于这个字典
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabe = classify(secondDict[key],featLabels,testVec)#递归向下寻找到非字典的情况，此时是叶子节点，叶子节点保存的肯定是类别
            else:
                classLabe=secondDict[key]#叶子节点，返回类别
    return classLabe

def test(mytree,labels,filename,sum,current,unreco):
    with open(filename, 'r')as csvfile:
        dataset=[line.strip().split(', ') for line in csvfile.readlines()]     #读取文件中的每一行
        dataset=[[int(i) if i.isdigit() else i for i in row] for row in dataset]    #对于每一行中的每一个元素，将行列式数字化并且去除空白保证匹配的正确完成
        cleanoutdata(dataset)   #数据清洗
        del(dataset[0])         #删除第一行和最后一行的空白数据
        del(dataset[-1])
        precondition(dataset)       #预处理数据集
        clean(dataset)          #把测试集中的，不存在于训练集中的数据清洗掉
        sum = len(dataset)
    for line in dataset:
        result=classify(mytree,labels,line)+'.'
        if result==line[8]:     #如果测试结果和类别相同
            current = current+1
        else :
            unreco = unreco + 1
    return sum,current,unreco

def precondition(mydate):#清洗连续型数据
    #continuous:0,2,4,10,11,12
    for each in mydate:
        del(each[0])
        del(each[1])
        del(each[2])
        del(each[7])
        del(each[7])
        del(each[7])

def clean(dataset):#清洗掉测试集中出现了训练集中没有的值的情况
    global mydate
    for i in range(8):
        set1=set()
        for row1 in mydate:
            set1.add(row1[i])
        for row2 in dataset:
            if row2[i] not in set1:
               dataset.remove(row2)
        set1.clear()

datasetname=r"C:\Users\yang\Desktop\adult.data"
mydate,label=createDateset(datasetname)
label_list=label[:]

Tree=createTree(mydate,label_list)

sum = 0
current = 0
unreco = 0

storetree(Tree,r'C:\Users\yang\Desktop\tree.txt') #保存决策树，避免下次再生成决策树

# Tree=grabTree(r'C:\Users\yang\Desktop\tree.txt')#读取决策树，如果已经存在tree.txt可以直接使用决策树不需要再次生成决策树

sum,current,unreco=test(Tree,label,r'C:\Users\yang\Desktop\adult.test',sum,current,unreco)
with open(r'C:\Users\yang\Desktop\trees.txt', 'w')as f:
    f.write(str(Tree))