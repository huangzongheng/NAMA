
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import torch
import numpy as np

datas = torch.load('../margin_to_dist.pt')
dist = datas['dist'] - datas['dist'].gather(1, indices[:, :10]).mean(-1, True)

data = dict()
data['a'] = datas['qg_margins'][~datas['is_pos']].numpy()[::10]
data['b'] = datas['dist'][~datas['is_pos']].numpy()[::10]

plt.figure(dpi=150)
plt.style.use('ggplot')
plt.scatter(data['a'],data['b'],c='steelblue',s=1, alpha=0.01)#绘制两组数据的散点图
plt.xlabel('b')
plt.ylabel('a')
plt.show()

# est = smf.ols(formula='a ~ b ', data=data).fit()
# x=data['b']#将a视为自变量
# y=data['a']
# est = sm.OLS(y, np.column_stack((x, y)))
# results = est.fit()
# # y_pred = est.predict(x)#最小二乘法求a的预测值
# # plt.plot(x, y_pred, c='r',linewidth=0.8,linestyle='--')#绘制下图红色拟合线
# print(results.params)#输出拟合系数
# print(results.summary())
# plt.show()