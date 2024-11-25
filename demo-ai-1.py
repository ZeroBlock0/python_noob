# 导入需要的库
from sklearn.linear_model import LinearRegression
import numpy as np

# 创建一些训练数据
# 这里我们创建身高(X)和体重(y)的数据
X = np.array([[150], [160], [170], [180]])  # 身高数据 (厘米)
y = np.array([45, 55, 65, 75])              # 体重数据 (公斤)

# 创建并训练模型
model = LinearRegression()
model.fit(X, y)

# 使用模型进行预测
height = 175  # 输入一个新的身高数据
weight_pred = model.predict([[height]])

# 打印结果
print(f"如果一个人身高{height}厘米，")
print(f"预测体重为：{weight_pred[0]:.1f}公斤")


#pip install scikit-learn numpy