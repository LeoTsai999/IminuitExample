import numpy as np                         # 引入numpy庫，主要用於數值計算
from iminuit import Minuit                 # 引入iminuit庫，主要用於最小化和擬合
from iminuit.cost import LeastSquares      # 引入iminuit.cost中的LeastSquares類，用於計算最小二乘擬合的成本函數
import matplotlib.pyplot as plt            # 引入matplotlib.pyplot庫，用於繪圖

# 定義模型函數
def line(x, a, b):
    return a * x + b

def sin_function(x, a, b):
    return a * np.sin(b * x)

# 生成數據
np.random.seed(42)  # 設置random seed
x_data = np.linspace(0, 10, 100)  # 每個點的x座標

# 直線數據（帶噪音）
a_true, b_true = 2, 1  # 直線真實參數(“理想上”的斜率和截距)
y_data_line = line(x_data, a_true, b_true) + np.random.normal(0, 0.5, x_data.size) # 帶有高斯分佈噪音的y座標

# 正弦波數據（帶噪音）
a_sin_true, b_sin_true = 5, 1  # 正弦波真實參數(""理想上”的振幅和波數)
y_data_sin = sin_function(x_data, a_sin_true, b_sin_true) + np.random.normal(0, 0.5, x_data.size) # 帶有高斯分佈噪音的y座標

# 定義誤差（假設每個數據點的誤差相同。誤差也可以是陣列/串列，但陣列尺寸就必須和x座標，y座標相同（這樣才有一一對應））
y_error = 0.5  # 高斯分佈噪音標準差

# 對直線進行擬合
cost_line = LeastSquares(x_data, y_data_line, y_error, line) # 聲明一個LeastSquares物件，計算擬合直線的成本函數（成本函數就是程式背後在最小化的東西，類似你高一學的最小平方法）
m_line = Minuit(cost_line, a=1.5, b=0.5)  # 給予初始猜測值
m_line.migrad()  # 執行擬合

# 對正弦波進行擬合
cost_sin = LeastSquares(x_data, y_data_sin, y_error, sin_function)
m_sin = Minuit(cost_sin, a=0.8, b=0.8)  # 給予初始猜測值
m_sin.migrad()  # 執行擬合

# 繪圖
plt.figure(figsize=(12, 5))

# 直線擬合圖
plt.subplot(1, 2, 1)
# plt.scatter(x_data, y_data_line, s=20, label="Data (Line)", alpha=0.6)
plt.errorbar(x_data, y_data_line, yerr=y_error, fmt='o', color='gray', alpha=0.5, label="Error")
plt.plot(x_data, line(x_data, *m_line.values), 'r-', label=f"Fit: a={m_line.values['a']:.2f}, b={m_line.values['b']:.2f}")
plt.title("Line Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# 正弦波擬合圖
plt.subplot(1, 2, 2)
# plt.scatter(x_data, y_data_sin, s=20, label="Data (Sine)", alpha=0.6)
plt.errorbar(x_data, y_data_sin, yerr=y_error, fmt='o', color='gray', alpha=0.5, label="Error")
plt.plot(x_data, sin_function(x_data, *m_sin.values), 'r-', label=f"Fit: a={m_sin.values['a']:.2f}, b={m_sin.values['b']:.2f}")
plt.title("Sine Wave Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()

# 輸出擬合參數
print("Line Fit Parameters:")
print(f"a = {m_line.values['a']:.2f} ± {m_line.errors['a']:.2f}") # .2f是格式化輸出，只輸出到兩位小數
print(f"b = {m_line.values['b']:.2f} ± {m_line.errors['b']:.2f}")
print("\nSine Fit Parameters:")
print(f"a = {m_sin.values['a']:.2f} ± {m_sin.errors['a']:.2f}")
print(f"b = {m_sin.values['b']:.2f} ± {m_sin.errors['b']:.2f}")