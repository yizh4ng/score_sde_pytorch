import torch
import matplotlib.pyplot as plt

# 设置参数
gamma = 1.0  # 摩擦系数
k = 1.0  # 势能函数中的弹性系数，用于控制势能的形状
beta = 1.0  # 温度的倒数
dt = 0.01  # 时间步长
steps = 10000  # 模拟步数

# 初始化
x = torch.tensor([0.0], requires_grad=True)  # 初始位置
v = torch.randn(1)  # 初始速度从标准正态分布中抽取
trajectory = []  # 存储位置的列表


# 潜在能量函数和其梯度
def U(x):
    return 0.5 * k * x ** 2


# 模拟Underdamped Langevin Dynamics
for _ in range(steps):
    trajectory.append(x.item())

    # 计算梯度
    potential = U(x)
    potential.backward()

    # 更新动力学
    v = v - gamma * v * dt - x.grad * dt + torch.sqrt(torch.tensor(2 * gamma / beta * dt)) * torch.randn_like(v)
    x.data += v * dt  # 使用 .data 来更新x，避免影响自动梯度跟踪

    # 清除梯度
    if x.grad is not None:
        x.grad.zero_()
    else:
        x.requires_grad_(True)  # 确保x重新获得梯度

# 绘图
plt.plot(trajectory)
plt.xlabel('Time step')
plt.ylabel('Position x')
plt.title('Sampling trajectory of ULD')
plt.show()