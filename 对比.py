import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Read the first CSV file
df1 = pd.read_csv('train_final1.csv')
Average_Loss1 = df1['Train F1'].values
df2 = pd.read_csv('train_final2.csv')
Average_Loss2 = df2['Train F1'].values
df3=pd.read_csv("train_final3.csv")
Average_Loss3=df3["Train F1"].values
# Extract the Epoch values for x-axis
epoch_values = df1['Epoch'].values

plt.plot(epoch_values, Average_Loss2, label='QLSTM Train F1')
plt.plot(epoch_values, Average_Loss1, label='LSTM Train F1')
plt.plot(epoch_values, Average_Loss3, label='Onlylayer LSTM Train F1')
plt.xlabel('Epoch')
plt.ylabel('Train F1')
plt.legend()
plt.title('Comparison of Train F1 between Models')
plt.show()
# #计算平均值
# train_recall1_mean = np.mean(train_recall1)
# train_recall2_mean = np.mean(train_recall2)
# train_recall3_mean = np.mean(train_recall3)
#
# # 计算标准差
# train_recall1_std = np.std(train_recall1)
# train_recall2_std = np.std(train_recall2)
# train_recall3_std = np.std(train_recall3)
#
# # 计算置信区间
# train_recall1_lower = train_recall1_mean - 1.96 * train_recall1_std / np.sqrt(len(train_recall1))
# train_recall1_upper = train_recall1_mean + 1.96 * train_recall1_std / np.sqrt(len(train_recall1))
#
# train_recall2_lower = train_recall2_mean - 1.96 * train_recall2_std / np.sqrt(len(train_recall2))
# train_recall2_upper = train_recall2_mean + 1.96 * train_recall2_std / np.sqrt(len(train_recall2))
#
# train_recall3_lower = train_recall3_mean - 1.96 * train_recall3_std / np.sqrt(len(train_recall3))
# train_recall3_upper = train_recall3_mean + 1.96 * train_recall3_std / np.sqrt(len(train_recall3))
#
# # 绘制带置信区间的折线图
# plt.plot(epoch_values, train_recall1, label='QLSTM Train Recall', color='blue')
# plt.fill_between(epoch_values, train_recall1_lower, train_recall1_upper, color='blue', alpha=0.2)
# plt.plot(epoch_values, train_recall2, label='LSTM Train Recall', color='red')
# plt.fill_between(epoch_values, train_recall2_lower, train_recall2_upper, color='red', alpha=0.2)
# # plt.plot(epoch_values, train_recall3, label='Onlylayer LSTM Train Recall', color='green')
# # plt.fill_between(epoch_values, train_recall3_lower, train_recall3_upper, color='green', alpha=0.2)
# plt.xlabel('Epoch')
# plt.ylabel('Precision')
# plt.legend()
# plt.title('Comparison of Train Recall with Confidence Intervals')
# plt.show()
#
# import pandas as pd
# import math
#
# # 加载CSV文件
# csv_file_path = 'train_metrics3.csv'
#
# df = pd.read_csv(csv_file_path)
#
#
# def calculate_precision(F1, recall):
#     """
#     根据F1分数和召回率计算精确率。
#     """
#     a = 2
#     b = -F1
#     c = -F1 * recall
#     discriminant = b ** 2 - 4 * a * c
#
#     if discriminant < 0:
#         print("方程无实数解")
#         return None
#
#     sol1 = (-b + math.sqrt(discriminant)) / (2 * a)
#     sol2 = (-b - math.sqrt(discriminant)) / (2 * a)
#
#     valid_solutions = [sol for sol in [sol1, sol2] if sol > 0]
#     return max(valid_solutions) if valid_solutions else None
#
# # 计算前10行的精确率并存储到列表中
# precision_values = []
# for i in range(10):  # 读取前10行
#     F1_example = df.loc[i, 'Train F1']
#     recall_example = df.loc[i, 'Train Recall']
#     precision = calculate_precision(F1_example, recall_example)
#     if precision is not None:
#         precision_values.append(precision)
#         print(f"第{i+1}行 - 根据训练F1分数({F1_example})和召回率({recall_example}), 计算得到的精确率为: {precision}")
#     else:
#         print(f"第{i+1}行 - 无法计算精确率，检查输入值({F1_example}, {recall_example})是否合理。")
#
# print("\n所有计算出的精确率值列表:", precision_values)
#
# #将precison存入csv文件
# df['Precision'] = precision_values
# df.to_csv(csv_file_path, index=False)