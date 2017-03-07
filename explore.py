# 为这个项目导入需要的库
import pandas as pd

# 导入附加的可视化代码visuals.py

# 导入人口普查数据
data = pd.read_csv("census.csv")

income = data['income']


# 成功 - 显示第一条记录
# display(data.head(n=1))

def greater(data):
    count = 0
    for _, person in data.iterrows():
        if '>50K' == person['income']:
            count += 1
    return count


def atMost(data):
    count = 0
    for _, person in data.iterrows():
        if '<=50K' == person['income']:
            count += 1
    return count


# 练习：数据探索
# 总的记录数
n_records = income.count()

# 被调查者的收入大于$50,000的人数
n_greater_50k = greater(data)

# 被调查者的收入最多为$50,000的人数
n_at_most_50k = atMost(data)

# 被调查者收入大于$50,000所占的比例
greater_percent = n_greater_50k * 100 / n_records

# 打印结果
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))
