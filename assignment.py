# coding: UTF-8
# 为这个项目导入需要的库
import numpy as np
import pandas as pd
from time import time
from IPython.display import display  # 允许为DataFrame使用display()
# 导入sklearn.preprocessing.StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# 从sklearn中导入两个评价指标 - fbeta_score和accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score
# 导入附加的可视化代码visuals.py
import visuals as vs
# 从sklearn中导入三个监督学习模型
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# 导入人口普查数据
data = pd.read_csv("census.csv")

# 练习：数据探索
n_records = float(len(data))

n_greater_50k = len(data[data.income == '>50K'])

n_at_most_50k = len(data[data.income == '<=50K'])

greater_percent = n_greater_50k * 100 / n_records

# Print the results
# print("Total number of records: {}".format(n_records))
# print("Individuals making more than $50,000: {}".format(n_greater_50k))
# print("Individuals making at most $50,000: {}".format(n_at_most_50k))
# print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))

# 转换倾斜的连续特征
# 将数据切分成特征和对应的标签
income_raw = data['income']
features_raw = data.drop('income', axis=1)

# 可视化原来数据的倾斜的连续特征
# vs.distribution(data)
# capital-gain ：资本收益
# capital-loss ：资本流失

# 对于倾斜的数据使用Log转换
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# 可视化经过log之后的数据分布
# vs.distribution(features_raw, transformed = True)

# 规一化数字特征
# 初始化一个 scaler，并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# 显示一个经过缩放的样例记录
# display(features_raw.head(n = 1))

# 练习：数据预处理
# 使用pandas.get_dummies()对'features_raw'数据进行独热编码
features = pd.get_dummies(features_raw)

# 将'income_raw'编码成数字值
income = income_raw.apply(lambda x: 1 if x == '>50K' else 0)

# 打印经过独热编码之后的特征数量
encoded = list(features.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# 移除下面一行的注释以观察编码的特征名字
# print(encoded)

# 混洗和切分数据
# 将'features'和'income'数据切分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size=0.2, random_state=0)

# 显示切分的结果
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

# 评价模型性能
# 问题 1 - 朴素预测器的性能
# 计算准确率，如果我们选择一个预测收入都大于50K的模型，那么这个模型的正确率为：
accuracy = greater_percent / 100

# 使用上面的公式，并设置beta=0.5计算F-score
beta = 0.5
recall = 1
fscore = (1 + beta ** 2) * accuracy * recall / (beta ** 2 * accuracy + recall)


# 打印结果
# print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# 问题 2 - 模型应用
# 练习 - 创建一个训练和预测的流水线
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    results = {}

    # 使用sample_size大小的训练数据来拟合学习器
    start = time()  # 获得程序开始时间
    learner.fit(X_train[0:int(sample_size)], y_train[0:int(sample_size)])
    end = time()  # 获得程序结束时间

    # 计算训练时间
    results['train_time'] = end - start

    #  得到在测试集上的预测值
    #  然后得到对前300个训练数据的预测结果
    start = time()  # 获得程序开始时间
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[0:300])
    end = time()  # 获得程序结束时间

    # 计算预测用时
    results['pred_time'] = end - start

    # 计算在最前面的300个训练数据的准确率
    results['acc_train'] = accuracy_score(y_train[0:300], predictions_train)

    # 计算在测试集上的准确率
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # 计算在最前面300个训练数据上的F-score
    results['f_train'] = fbeta_score(y_train[0:300], predictions_train, beta=0.5)

    # 计算测试集上的F-score
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)

    # 成功
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    # 返回结果
    return results


# 练习：初始模型的评估
# 初始化三个模型
clf_A = LogisticRegression(random_state=0)
clf_B = LinearSVC(random_state=0)
clf_C = GradientBoostingClassifier(random_state=0)

# 计算1%， 10%， 100%的训练数据分别对应多少点
samples_1 = int(len(y_train) * 0.01)
samples_10 = int(len(y_train) * 0.1)
samples_100 = len(y_train)

# 收集学习器的结果
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
            train_predict(clf, samples, X_train, y_train, X_test, y_test)

# 对选择的三个模型得到的评价结果进行可视化
vs.evaluate(results, accuracy, fscore)
