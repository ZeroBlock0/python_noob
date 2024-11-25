import tensorflow as tf
import numpy as np

# 准备训练数据
heights = np.array([150, 160, 170, 180], dtype=float)  # 身高数据
weights = np.array([45, 55, 65, 75], dtype=float)      # 体重数据

# 创建一个最简单的神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

# 编译模型
model.compile(optimizer='sgd', loss='mse')

# 训练模型
model.fit(heights, weights, epochs=100, verbose=0)

# 进行预测
height = 175
prediction = model.predict([height])

# 打印结果
print(f"如果一个人身高{height}厘米，")
print(f"预测体重为：{prediction[0][0]:.1f}公斤")