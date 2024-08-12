import matplotlib.pyplot as plt
from hist_mnist import *
# 收集 x 轴与 y 轴的数据
# x = list(range(42, 525+21, 21))
x = list(range(20, 120, 20))
x.insert(0,10)

ddpm = [float(x) for x in ddpm.strip().split()]
# langevin = [float(x) for x in langevin.strip().split()]
# mala = [float(x) for x in mala.strip().split()]
mala_es = [float(x) for x in mala_es.strip().split()]
uld = [float(x) for x in uld.strip().split()]
# ald = [float(x) for x in ald.strip().split()]
# ddim = [float(x) for x in ddim.strip().split()]
# 绘制折线图
plt.plot(x, ddpm, color='blue', linewidth=3, marker='o', label='DDPM')
# plt.plot(x, ddim, color='black', linewidth=3, marker='o', label='DDIM')
# plt.plot(x, ald, color='purple', linewidth=3, marker='o', label='ALD')
# plt.plot(x, langevin, color='orange', linewidth=3, marker='o', label='ULA')
plt.plot(x, uld, color='green', linewidth=3, marker='o', label='ULD')
plt.plot(x, mala_es, color='brown', linewidth=3, marker='o', label='MALA_ES')
# plt.plot(x, mala, color='red', linewidth=3, marker='o', label='MALA')

# 坐标轴上负号的正常显示
# plt.rcParams["axes.unicode_minus"] = False
# 添加标题和坐标轴标签
# plt.title('漂亮的折线图', fontsize=20)
plt.xlabel('Number of Function Evaluations', fontsize=16)
plt.ylabel('FID', fontsize=16)
# 显示图例
plt.legend(fontsize=14)
# 添加网格线
plt.grid(True)
# 显示图形
plt.savefig('mnist.png', dpi=600)
plt.show()
plt.close()
