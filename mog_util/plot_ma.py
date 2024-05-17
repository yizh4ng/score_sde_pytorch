import matplotlib.pyplot as plt
# 收集 x 轴与 y 轴的数据
x = list(range(42, 525+21, 21))
ddpm = """
0.869355023
0.889720798
0.903890789
0.914193988
0.923172891
0.929006636
0.934248984
0.937836051
0.942075014
0.944604635
0.947159231
0.949908435
0.951726973
0.95391643
0.956447899
0.957267821
0.958364844
0.959836423
0.961127996
0.961613238
0.963079512
0.964567006
0.96445334
0.966418386

"""
langevin = """
0.975880802
0.973004162
0.970600009
0.968462825
0.967529714
0.966341496
0.965197921
0.964628458
0.964616895
0.965408742
0.965800166
0.965403736
0.965951025
0.966908753
0.967529714
0.967486501
0.968297124
0.968531013
0.969610929
0.969523728
0.970297277
0.970254719
0.971092343
0.972346187

"""

mala = """
0.975146532
0.971825004
0.970739722
0.96869719
0.9676193
0.96788764
0.967007756
0.967928648
0.967254043
0.968209326
0.968685448
0.968593955
0.969193935
0.970316827
0.970519543
0.970436096
0.971318364
0.970897973
0.972200274
0.97261554
0.972640753
0.972305536
0.972524524
0.973249972
"""

ddpm = [float(x) for x in ddpm.strip().split()]
langevin = [float(x) for x in langevin.strip().split()]
mala = [float(x) for x in mala.strip().split()]
# 绘制折线图
plt.plot(x, ddpm, color='blue', linewidth=3, marker='o', label='DDPM')
plt.plot(x, langevin, color='orange', linewidth=3, marker='o', label='Langevin')
plt.plot(x, mala, color='red', linewidth=3, marker='o', label='MALA')

# 坐标轴上负号的正常显示
# plt.rcParams["axes.unicode_minus"] = False
# 添加标题和坐标轴标签
# plt.title('漂亮的折线图', fontsize=20)
plt.xlabel('Number of Function Evaluations', fontsize=16)
plt.ylabel('Marginal Accuracy', fontsize=16)
# 显示图例
plt.legend(fontsize=14)
# 添加网格线
plt.grid(True)
# 显示图形
plt.savefig('meanshift.png', dpi=600)
plt.show()
plt.close()
