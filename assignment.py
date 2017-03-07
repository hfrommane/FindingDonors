# 为这个项目导入需要的库
import numpy as np
import pandas as pd
from time import time
from IPython.display import display  # 允许为DataFrame使用display()

# 导入附加的可视化代码visuals.py
import visuals as vs

# 导入人口普查数据
data = pd.read_csv("census.csv")

income = data['income']
