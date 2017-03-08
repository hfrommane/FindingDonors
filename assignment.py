# coding: UTF-8
# 为这个项目导入需要的库
import numpy as np
import pandas as pd
from time import time
# 允许为DataFrame使用display()
from IPython.display import display
# 导入sklearn.preprocessing.StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# 从sklearn中导入两个评价指标 - fbeta_score和accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score
# 导入附加的可视化代码visuals.py
import visuals as vs
# 从sklearn中导入三个监督学习模型
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.base import clone

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
# vs.evaluate(results, accuracy, fscore)

# 问题 3 - 选择最佳的模型
# 梯度增强分类器模型（GBC）是我的第一选择。上述测试的结果显示GBC具有最高的准确性和fbeta得分，
# 由于我以前使用虚拟变量格式化数据集，因此使用决策树预测二进制输出是合理的。
# 此外，虽然逻辑回归似乎是一个可行的候选人，有更多的选择调整GBC模型参数以后在项目中。

# 问题 4 - 用通俗的话解释模型
# GBC模型是从常见的决策树创建的。这些“树”是算法内的决策叉，它基于当前被“树”提出的输入来驱动输出。
# 在做出“决定”之后，输出被跟随到另一个分支，其中询问和分析另一个问题或数据的参数，并且该过程继续，
# 直到达到最终输出，在这种情况下是否正在分析输入个人的迹象表明收入超过50K。

# 这是一个决策树。梯度增强来自于一遍又一遍地重复这种迭代，更加强调收入被错误估计的人。
# 这样，模型通过这些多次迭代来训练自己，以了解什么数据难以分类，什么不是，每次都给自己一点线索。
# 随着时间的推移，模型结合了这些小小的线索，以了解数据的真实性质，即使它可能看起来很复杂的用户或客户。

# 实施：模型校正
# 初始化分类器
clf = GradientBoostingClassifier(random_state=0)

# 创建你希望调节的参数列表
parameters = {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.10, 0.15], 'max_depth': range(2, 7)}

# 创建一个fbeta_score打分对象
scorer = make_scorer(fbeta_score, beta=0.5)

cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

# 在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# 用训练数据拟合网格搜索对象并找到最佳参数
grid_fit = grid_obj.fit(X_train, y_train)

# 得到estimator
best_clf = grid_fit.best_estimator_

# 使用没有调优的模型做预测
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# 调优参数
print("Parameter 'n_estimators' is {} for the optimal model.".format(best_clf.get_params()['n_estimators']))
print("Parameter 'learning_rate' is {} for the optimal model.".format(best_clf.get_params()['learning_rate']))
print("Parameter 'max_depth' is {} for the optimal model.".format(best_clf.get_params()['max_depth']))
# 汇报调参前和调参后的分数
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5)))

# 问题 5 - 最终模型评估
# 优化的值比未优化的模型值更好，并且都远高于基准预测器。找到optomized模型花费了可观的时间量，
# 但提高模型的准确性甚至一个百分点似乎值得的时间成本和洞察模型。虽然，如果我试图优化所有的监督学习技术，我看到时间成本可以开始快速堆叠。

# 问题 6 - 观察特征相关性
# 练习 - 提取特征重要性
# 在训练集上训练一个监督学习模型
model = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)

# 提取特征重要性
importances = model.feature_importances_

# 绘图
# vs.feature_plot(importances, X_train, y_train)

# 问题 7 - 提取特征重要性

# 特征选择
# 导入克隆模型的功能


# 减小特征空间
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# 在前面的网格搜索的基础上训练一个“最好的”模型
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# 做一个新的预测
reduced_predictions = clf.predict(X_test_reduced)

# 对于每一个版本的数据汇报最终模型的分数
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta=0.5)))

# 问题 8 - 特征选择的影响
# 缩减模型的精度为-3％，与全数据训练模型相比，其fbeta为-0.05。如果训练时间是一个因素，
# 这可能是一个可能的解决方案，但如果是这种情况，可能值得切换到逻辑回归模型。
# 从本报告的早期的图表，其训练和预测的时间都几乎是瞬时的，具有更高的准确性和fbeta分数比降低特征梯度增强分类器模型。