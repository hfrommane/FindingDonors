# FindingDonors
**这个项目只能用Python2**
### 数据

修改的人口普查数据集含有将近32,000个数据点，每一个数据点含有13个特征。这个数据集是Ron Kohavi的论文*"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",*中数据集的一个修改版本。你能够在[这里](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf)找到论文，在[UCI的网站](https://archive.ics.uci.edu/ml/datasets/Census+Income)找到原始数据集。

**特征**

- `age`: 一个整数，表示被调查者的年龄。
- `workclass`: 一个类别变量表示被调查者的通常劳动类型，允许的值有 {Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked}
- `education_level`: 一个类别变量表示教育程度，允许的值有 {Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool}
- `education-num`: 一个整数表示在学校学习了多少年
- `marital-status`: 一个类别变量，允许的值有 {Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse}
- `occupation`: 一个类别变量表示一般的职业领域，允许的值有 {Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces}
- `relationship`: 一个类别变量表示家庭情况，允许的值有 {Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried}
- `race`: 一个类别变量表示人种，允许的值有 {White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black}
- `sex`: 一个类别变量表示性别，允许的值有 {Female, Male}
- `capital-gain`: 连续值。
- `capital-loss`: 连续值。
- `hours-per-week`: 连续值。
- `native-country`: 一个类别变量表示原始的国家，允许的值有 {United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands}

**目标变量**

- `income`: 一个类别变量，表示收入属于那个类别，允许的值有 {<=50K, >50K}

## 练习：数据探索
	n_records = float(len(data))
	
	n_greater_50k = len(data[data.income == '>50K'])
	
	n_at_most_50k = len(data[data.income == '<=50K'])
	
	greater_percent = n_greater_50k * 100 / n_records
	
	# Print the results
	print("Total number of records: {}".format(n_records))
	print("Individuals making more than $50,000: {}".format(n_greater_50k))
	print("Individuals making at most $50,000: {}".format(n_at_most_50k))
	print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))

程序运行结果：

- Total number of records: 45222.0
- Individuals making more than $50,000: 11208
- Individuals making at most $50,000: 34014
- Percentage of individuals making more than $50,000: 24.78%

## 转换倾斜的连续特征
	# 将数据切分成特征和对应的标签
	income_raw = data['income']
	features_raw = data.drop('income', axis = 1)
	
	# 可视化原来数据的倾斜的连续特征
	vs.distribution(data)
程序运行结果：
![](https://raw.githubusercontent.com/hfrommane/FindingDonors/master/figure/figure_1.png)

	# 对于倾斜的数据使用Log转换
	skewed = ['capital-gain', 'capital-loss']
	features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))
	
	# 可视化经过log之后的数据分布
	vs.distribution(features_raw, transformed = True)
程序运行结果：
![](https://raw.githubusercontent.com/hfrommane/FindingDonors/master/figure/figure_2.png)

## 规一化数字特征
	# 导入sklearn.preprocessing.StandardScaler
	from sklearn.preprocessing import MinMaxScaler
	
	# 初始化一个 scaler，并将它施加到特征上
	scaler = MinMaxScaler()
	numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
	features_raw[numerical] = scaler.fit_transform(data[numerical])
	
	# 显示一个经过缩放的样例记录
	display(features_raw.head(n = 1))

## 练习：数据预处理（one-hot encoding）
	# 使用pandas.get_dummies()对'features_raw'数据进行独热编码
	features = pd.get_dummies(features_raw)
	
	# 将'income_raw'编码成数字值
	income = income_raw.apply(lambda x: 1 if x == '>50K' else 0)
	
	# 打印经过独热编码之后的特征数量
	encoded = list(features.columns)
	print("{} total features after one-hot encoding.".format(len(encoded)))
	
	# 移除下面一行的注释以观察编码的特征名字
	# print(encoded)
程序运行结果：

- 103 total features after one-hot encoding.

## 混洗和切分数据
	# 将'features'和'income'数据切分成训练集和测试集
	X_train, X_test, y_train, y_test = train_test_split(features, income, test_size=0.2, random_state=0)
	
	# 显示切分的结果
	print("Training set has {} samples.".format(X_train.shape[0]))
	print("Testing set has {} samples.".format(X_test.shape[0]))
程序运行结果：

- Training set has 36177 samples.
- Testing set has 9045 samples.

## 问题 1 - 朴素预测器的性能
	# 计算准确率，如果我们选择一个预测收入都大于50K的模型，那么这个模型的正确率为：
	accuracy = greater_percent / 100
	
	# 使用上面的公式，并设置beta=0.5计算F-score
	beta = 0.5
	recall = 1
	fscore = (1 + beta ** 2) * accuracy * recall / (beta ** 2 * accuracy + recall)
程序运行结果：

Naive Predictor: [Accuracy score: 0.2478, F-score: 0.2917]

## 问题 2 - 模型应用
- 支撑向量机 (SVM)
    - 应用场景：客户分类模型
    - 优点：
        - 用于分类，回归和异常值检测
        - 在高维空间有效
        - 在决策函数中使用训练点的子集，因此它也具有记忆效率
        - 有效执行线性分类
        - 可以使用内核技巧执行非线性分类
    - 缺点：
        - 如果特征的数目远大于样本的数目，则该方法性能可能很差
        - SVM不直接提供概率估算，这些是使用五折交叉验证计算的
    - 为什么这个模型适合这个问题：数据具有高维空间和足够的样本，使SVM是一个合理的选择。
- Logistic回归
    - 应用场景：业务，客户保留
    - 优点：
        - 适用于预测分类结果
        - 如果有单个决策边界，不一定平行于轴，工作得更好
        - 有助于理解几个独立变量对结果变量的影响
        - 当关注一个适当的因变量时，逻辑回归往往是一个很好的替代其他工具的模型
    - 缺点：
        - 如果数据中包括非信息特征，则该模型将具有很少至没有预测值
        - 不能精确预测连续结果
        - 每个数据点必须独立于所有其他数据点
        - 高偏差
    - 为什么这个模型适合这个问题：目标数据与许多独立特征是二分的。
- 梯度增强分类器（决策树）
    - 应用场景：商业管理
    - 优点：
         - 结合多个弱学习者，这可以均衡地导致更鲁棒的模型
         - 易于理解，可以在视觉上很少解释
    - 缺点：
        - 随着模型复杂度的增加，模型趋向于越不准确
        - 没有适当的“修剪”，他们有过度的倾向
        - 随着越来越多的功能被添加，训练时间会快速增加
    - 为什么这个模型适合这个问题：
        - 数据已经结构化了所有功能的虚拟变量，因此它是以正确的形式，以分解树的形式提出分层问题。此外，随着这么多不同的特性的发挥，增强的整体模型将有助于看透数据的模糊性。

## 练习 - 创建一个训练和预测的流水线
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

## 练习：初始模型的评估
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
程序运行结果：

- LogisticRegression trained on 361 samples.
- LogisticRegression trained on 3617 samples.
- LogisticRegression trained on 36177 samples.
- LinearSVC trained on 361 samples.
- LinearSVC trained on 3617 samples.
- LinearSVC trained on 36177 samples.
- GradientBoostingClassifier trained on 361 samples.
- GradientBoostingClassifier trained on 3617 samples.
- GradientBoostingClassifier trained on 36177 samples.
![](https://raw.githubusercontent.com/hfrommane/FindingDonors/master/figure/figure_3.png)

## 问题 3 - 选择最佳的模型
梯度增强分类器模型（GBC）是我的第一选择。上述测试的结果显示GBC具有最高的准确性和fbeta得分，由于我之前使用了虚拟变量格式化数据集，因此使用决策树预测二进制输出是合理的。

## 问题 4 - 用通俗的话解释模型
GBC模型是从常见的决策树创建的。这些“树”是算法内的决策叉，它基于当前被“树”提出的输入来驱动输出。在做出“决定”之后，输出被跟随到另一个分支，其中询问和分析另一个问题或数据的参数，并且该过程继续，直到达到最终输出，在这种情况下是否正在分析输入个人的迹象表明收入超过50K。

这是一个决策树。梯度增强来自于一遍又一遍地重复这种迭代，更加强调收入被错误估计的人。这样，模型通过这些多次迭代来训练自己，以了解什么数据难以分类，什么不是，每次都给自己一点线索。随着时间的推移，模型结合了这些小小的线索，以了解数据的真实性质，即使它可能看起来很复杂的用户或客户。

## 练习：模型调优
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
程序运行结果：

- Parameter 'n_estimators' is 50 for the optimal model.
- Parameter 'learning_rate' is 0.15 for the optimal model.
- Parameter 'max_depth' is 6 for the optimal model.

- Unoptimized model
- Accuracy score on testing data: 0.8630
- F-score on testing data: 0.7395

- Optimized Model
- Final accuracy score on the testing data: 0.8701
- Final F-score on the testing data: 0.7516

## 问题 5 - 最终模型评估
    |  评价指标  | 基准预测器 | 未优化的模型 | 优化的模型 |
    | :-------: | :-------: | :---------: | :-------: |
    |   准确率   |   0.2478  |    0.8630   |   0.8701  |
    |  F-score  |   0.2917  |    0.7395   |   0.7516  |
优化的值比未优化的模型值更好，并且都远高于基准预测器。找到optomized模型花费了不少时间，但提高模型的准确性甚至一个百分点都值得花费一些时间成本。

## 问题 6 - 观察特征相关性
## 练习 - 提取特征重要性
	# 在训练集上训练一个监督学习模型
	model = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
	
	# 提取特征重要性
	importances = model.feature_importances_
	
	# 绘图
	vs.feature_plot(importances, X_train, y_train)
![](https://raw.githubusercontent.com/hfrommane/FindingDonors/master/figure/figure_4.png)

## 问题 7 - 提取特征重要性
## 特征选择
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
程序运行结果：

- Final Model trained on **full data**
- Accuracy on testing data: 0.8701
- F-score on testing data: 0.7516
- Final Model trained on **reduced data**
- Accuracy on testing data: 0.8593
- F-score on testing data: 0.7259

## 问题 8 - 特征选择的影响
与全数据训练模型相比，缩减模型的精度为-1％，其fbeta为-0.05。如果训练时间是一个因素，这可能是一个可能的解决方案。但如果是这种情况，可以切换到逻辑回归模型，从本报告的早期的图表，其训练和预测的时间都几乎是瞬时的，具有更高的准确性和fbeta分数比降低特征的梯度增强分类器（决策树）更好。