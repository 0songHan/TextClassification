from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from icecream import ic






#数据获取
data = pd.read_csv('../data/train_new.csv')
copus = data['words'].values
print(copus)

#特征工程
stop_words = open('../data/stopwords.txt',encoding='utf-8').read().split()
# print(stop_words)
tfidf = TfidfVectorizer(stop_words=stop_words)
text_vectors = tfidf.fit_transform(copus)
# print(text_vectors)

#模型构建
rf = RandomForestClassifier()
label = data['label']
X_train, X_test, y_train, y_test = train_test_split(text_vectors,label,test_size=0.2)
rf.fit(X_train,y_train)

#模型预测
y_pred = rf.predict(X_test)

#模型评估
acc = accuracy_score(y_pred,y_test)
print(acc)

