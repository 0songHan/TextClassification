import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import jieba

data = pd.read_csv('../data/dev.txt',sep='\t',names=['sentence','label'])#制表符进行划分，默认是， 所以要指明
# print(data.head(10))
# print(len(data))

# print(Counter(data['label'].values))#统计标签的数量
# print(len(Counter(data['label'].values)))
count = Counter(data['label'].values)

#分析样本数量的分布
#（为了获得长句子）一般均值加上1-3倍方差，不会加上4倍，也可以不加/还可以中位数/众数等方法

#比例统计
total = len(data)


for key,value in count.items():#这里的items才是可以迭代的count 是一个 Counter 对象，它不能直接用于迭代解包。
    # 正确的方式是使用 Counter 对象的 items() 方法来获取键值对，并进行遍历
    #比例用饼图
    ratio = value/total
    # print(key,ratio*100,'%')
label_ratios = {label:count/total for label,count in count.items()}

plt.figure(figsize=(8, 6))
plt.pie(label_ratios.values(), labels=label_ratios.keys(), autopct='%1.1f%%')
plt.title('Label Distribution')
plt.show()

#句子（字）长度统计
data['sentence_len'] = data['sentence'].apply(len)
# print(data.head(10))

#均值和方差统计（最多使用来设置句子长度）
length_mean = np.mean(data['sentence_len'])
length_std = np.std(data['sentence_len'])

#分词（结巴分词）
def cut_sentence(s):
    list_word = list(jieba.cut(s))#这里不用list就是个对象，lcut返回的就是列表
    # list_word = jieba.cut(s)
    return list_word
#d对data中的数据进行处理
# data['words'] = data['sentence'].apply(cut_sentence)
# print(data.head(10))这里返回列表

# data['words'] = data['sentence'].apply(lambda s : ' '.join(cut_sentence(s)))
# print(data.head(10))这里给返回字符串
# data['words'] = data['words'].apply(lambda s:' '.join(s.split())[:30])#这里要连接成句子然后再切分，因为上面统计的是字的长度
# print(data.head(10))

#保存
# data.to_csv('../data/dev_new.csv')






