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

#计算香农熵/期望信息
def calculateEntropy(dataSet):
    ClassifyCount = {}#分类标签统计字典，用来统计每个分类标签的概率
    for vector in dataSet:
        clasification = vector[-1]  #获取分类
        if not clasification not in ClassifyCount.keys():#如果分类暂时不在字典中，在字典中添加对应的值对
            ClassifyCount[clasification] = 0
        ClassifyCount[clasification] += 1         #计算出现次数
    shannonEntropy=0.0
    for key in ClassifyCount:
        probability=float(ClassifyCount[key]) / dataSet.shape[0]      #计算概率
        shannonEntropy -= probability * log(probability,2)   #香农熵的每一个子项都是负的
    return shannonEntropy

# def addFetureValue(feature):

#划分数据集
def splitDataSet(dataSet,featureIndex,value):
    newDataSet=[]
    for vec in dataSet:#将选定的feature的列从数据集中去除
        if vec[featureIndex] == value:
            rest = vec[:featureIndex]
            rest.extend(vec[featureIndex + 1:])
            newDataSet.append(rest)
    return newDataSet


def addFeatureValue(featureListOfValue,feature):
    feat = [[ 'Private', 'Self-emp-not-inc', 'Self-emp-inc',
              'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
            [],[],[],[],[]]
    for featureValue in feat[feature]: #feat保存的是所有属性特征的所有可能的取值，其结构为feat = [ [val1,val2,val3,…,valn], [], [], [], … ,[] ]
        featureListOfValue.append(featureValue)

#选择最好的数据集划分方式
def chooseBestSplitWay(dataSet):
    HC = calculateEntropy(dataSet)#计算整个数据集的香农熵(期望信息)，即H(C)，用来和每个feature的香农熵进行比较
    bestfeatureIndex = -1                   #最好的划分方式的索引值，因为0也是索引值，所以应该设置为负数
    gain=0.0                        #信息增益=期望信息-熵，gain为最好的信息增益，IG为各种划分方式的信息增益
    for feature in range(len(dataSet[0]) -1 ): #计算feature的个数，由于dataset中是包含有类别的，所以要减去类别
        featureListOfValue=[vector[feature] for vector in dataSet] #对于dataset中每一个feature，创建单独的列表list保存其取值，其中是不重复的
        addFeatureValue(featureListOfValue,feature) #增加在训练集中有，测试集中没有的属性特征的取值
        unique=set(featureListOfValue)
        HTC=0.0         #保存HTC，即H（T|C）
        for value in unique:
            subDataSet = splitDataSet(dataSet,feature,value)  #划分数据集
            probability = len(subDataSet) / float(len(dataSet))  #求得当前类别的概率
            HTC += probability * calculateEntropy(subDataSet)      #计算当前类别的香农熵，并和HTC想加，即H(T|C) = H（T1|C）+ H(T2|C) + … + H(TN|C)
        IG=HC-HTC        #计算对于该种划分方式的信息增益
        if(IG > gain):
            gain = IG
            bestfeatureIndex = feature
    return bestfeatureIndex

#返回出现次数最多的类别，避免产生所有特征全部用完无法判断类别的情况
def majority(classList):
    classificationCount = {}
    for i in classList:
        if not i in classificationCount.keys():
            classificationCount[i] = 0
        classificationCount[i] += 1
    sortedClassification = sorted(dict2list(classificationCount),key = operator.itemgetter(1),reverse = True)
    return sortedClassification[0][0]

#dict字典转换为list列表
def dict2list(dic:dict):
    keys=dic.keys()
    values=dic.values()
    lst=[(key,value)for key,value in zip(keys,values)]
    return lst

#创建树
def createTree(dataSet,labels):
    classificationList = [feature[-1] for feature in dataSet] #产生数据集中的分类列表，保存的是每一行的分类
    if classificationList.count(classificationList[0]) == len(classificationList): #如果分类别表中的所有分类都是一样的，则直接返回当前的分类
        return classificationList[0]
    if len(dataSet[0]) == 1: #如果划分数据集已经到了无法继续划分的程度，即已经使用完了全部的feature，则进行决策
        return majority(classificationList)
    bestFeature = chooseBestSplitWay(dataSet) #计算香农熵和信息增益来返回最佳的划分方案，bestFeature保存最佳的划分的feature的索引
    bestFeatureLabel = labels[bestFeature] #取出上述的bestfeature的具体值
    Tree = {bestFeatureLabel:{}}
    del(labels[bestFeature]) #删除当前进行划分是使用的feature避免下次继续使用到这个feature来划分
    featureValueList = [feature[bestFeature]for feature in dataSet] #对于上述取出的bestFeature,取出数据集中属于当前feature的列的所有的值
    uniqueValue = set(featureValueList) #去重
    for value in uniqueValue: #对于每一个feature标签的value值，进行递归构造决策树
        subLabels = labels[:]
        Tree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels)
    return Tree

def storeTree(inputree,filename):
    fw = open(filename, 'wb')
    pickle.dump(inputree, fw)
    fw.close()

def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

#测试算法
def classify(inputTree,featLabels,testVector):
    root = list(inputTree.keys())[0] #取出树的第一个标签,即树的根节点
    dictionary = inputTree[root] #取出树的第一个标签下的字典
    featIndex = featLabels.index(root)
    for key in dictionary.keys():#对于这个字典
        if testVector[featIndex] == key:
            if type(dictionary[key]).__name__ == 'dict': #如果还有一个新的字典
                classLabel = classify(dictionary[key],featLabels,testVector)#递归向下寻找到非字典的情况，此时是叶子节点，叶子节点保存的肯定是类别
            else:
                classLabel=dictionary[key]#叶子节点，返回类别
    return classLabel

def test(mytree,labels,filename,sum,correct,error):
    with open(filename, 'r')as csvfile:
        dataset=[line.strip().split(', ') for line in csvfile.readlines()]     #读取文件中的每一行
        dataset=[[int(i) if i.isdigit() else i for i in row] for row in dataset]    #对于每一行中的每一个元素，将行列式数字化并且去除空白保证匹配的正确完成
        cleanoutdata(dataset)   #数据清洗
        del(dataset[0])         #删除第一行和最后一行的空白数据
        del(dataset[-1])
        precondition(dataset)       #预处理数据集
        # clean(dataset)          #把测试集中的，不存在于训练集中的数据清洗掉
        sum = len(dataset)
    for line in dataset:
        result=classify(mytree,labels,line)+'.'
        if result==line[8]:     #如果测试结果和类别相同
            correct = correct + 1
        else :
            error = error + 1

    return sum,correct,error

def precondition(mydate):#清洗连续型数据
    #continuous:0,2,4,10,11,12
    for each in mydate:
        del(each[0])
        del(each[1])
        del(each[2])
        del(each[7])
        del(each[7])
        del(each[7])

# def clean(dataset):#清洗掉测试集中出现了训练集中没有的值的情况
#     global mydate
#     for i in range(8):
#         set1=set()
#         for row1 in mydate:
#             set1.add(row1[i])
#         for row2 in dataset:
#             if row2[i] not in set1:
#                dataset.remove(row2)
#         set1.clear()

dataSetName=r"C:\Users\yang\Desktop\adult.data"
mydate,label=createDateset(dataSetName)
labelList=label[:]

Tree=createTree(mydate,labelList)

sum = 0
correct = 0
error = 0

storeTree(Tree,r'C:\Users\yang\Desktop\tree.txt') #保存决策树，避免下次再生成决策树

# Tree=grabTree(r'C:\Users\yang\Desktop\tree.txt')#读取决策树，如果已经存在tree.txt可以直接使用决策树不需要再次生成决策树
sum,current,unreco=test(Tree,label,r'C:\Users\yang\Desktop\adult.test',sum,correct,error)
# with open(r'C:\Users\yang\Desktop\trees.txt', 'w')as f:
#     f.write(str(Tree))
print("准确率：%f" % correct / sum)