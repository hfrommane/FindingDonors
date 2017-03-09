# FindingDonors
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

## 练习：数据预处理
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
- 支持向量分类器
    - 行业：银行，确定公司的偿付能力
        - [支持向量机作为偿付能力分析的技术](https://core.ac.uk/download/pdf/6302770.pdf)
        - “...开发一种功能，通过对其分数值进行基准化，可以准确地分离溶剂和无力偿债公司的空间。
    - 优点：
        - 用于分类，回归和异常值检测
        - 在高维空间有效
        - 在决策函数中使用训练点的子集，因此它也具有记忆效率
        - 固有地执行线性分类
        - 可以使用内核技巧执行非线性分类
    - 缺点：
        - 如果特征的数目远大于样本的数目，则该方法可能给出差的性能
        - SVM不直接提供概率估计，这些是使用费用五折交叉验证计算的
    - 推理：数据具有高维空间和足够的样本，使SVM是一个合理的选择。
    
    
- 逻辑回归分类器
    - 行业：业务，客户保留
        - [使用逻辑回归预测客户保留](http://www.lexjansen.com/nesug/nesug98/solu/p095.pdf)
        - “适当应用后，逻辑回归模型可以产生强有力的洞察，为什么一些客户离开和其他人留下来。这些见解可以用于修改组织战略和/或评估实施这些战略的影响。
    - 优点：
        - 适用于预测分类结果
        - 如果有单个决策边界，不一定平行于轴，工作得更好
        - 有用于理解几个独立变量对结果变量的影响
        - 当适当关注创建一个适当的因变量时，逻辑回归往往是一个优越的（实质和统计）替代其他工具可用于模型事件结果
    - 缺点：
        - 如果数据中包括非信息特征，则该模型将具有很少至没有预测值
        - 不能精确预测连续结果
        - 每个数据点必须独立于所有其他数据点，否则模型将倾向于超重那些意见的重要性
        - 高偏差
    - 推理：目标数据与许多独立特征是二分的。
    
    
- 梯度增强分类器（决策树）
    - 行业：商业管理
        - [决策树](https://hbr.org/1964/07/decision-trees-for-decision-making)
    - 优点：
         - 结合多个弱学习者，这可以均衡地导致更鲁棒的模型
         - 易于解释，可以在视觉上很少解释
    - 缺点：
        - 随着模型复杂性的增加，它们趋向于越不准确
        - 没有适当的“修剪”，他们有过度的倾向
        - 随着越来越多的功能被添加，训练时间可以快速增加
    - 推理：
        - 数据已经结构化了所有功能的虚拟变量，因此它是以正确的形式，以分解树的形式提出分层问题。此外，随着这么多不同的特性的发挥，增强的整体模型将有助于看透数据的模糊性。