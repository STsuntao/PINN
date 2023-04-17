"""
A scratch for PINN solving the following PDE
Author: suntao
Date: 2023/4/15
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


epochs = 10000    # 训练代数
h = 100     # 画图网格密度
N = 1024    # 内点配置点数
N1 = 256    # 边界条件配置点数
N2 = 300    # 初始条件配置点数

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
# setup_seed(888888)

# 均匀取样
m = 34
n = 34
x = np.linspace(0, 1, m)
y = np.linspace(0, 1, n)
data = []
for i in range(1, m-1):
    for j in range(1, n-1):
        a = [x[i], y[j]]
        data.append(a)
data = torch.Tensor(data)
print(data.shape)
# Domain and Sampling
# def interior(n=N):
#     # 内点
#     x = torch.rand(n, 1)
#     t = torch.rand(n, 1)
#     cond = torch.exp(-(2*x-1)**2)*(1-torch.exp(-10*t))
#     return x.requires_grad_(True), t.requires_grad_(True), cond

def interior():
    # 内点，空间均匀采样
    x1 = data[:,0]
    x = x1.reshape(-1,1)
    t1 = data[:,1]
    t = t1.reshape(-1,1)
    cond = torch.exp(-(2*x-1)**2)*(1-torch.exp(-10*t))
    return x.requires_grad_(True), t.requires_grad_(True), cond

def left(n=N1):
    # 左边界
    t = torch.rand(n, 1)
    x = torch.zeros_like(t)
    cond = torch.zeros_like(t)
    return x.requires_grad_(True), t.requires_grad_(True), cond

def right(n=N1):
    # 右边界
    t = torch.rand(n, 1)
    x = torch.ones_like(t)
    cond = torch.zeros_like(t)
    return x.requires_grad_(True), t.requires_grad_(True), cond

def down(n=N2):
    # 初始条件
    x = torch.rand(n, 1)
    t = torch.zeros_like(x)
    cond = torch.ones_like(t)
    return x.requires_grad_(True), t.requires_grad_(True), cond

# Neural Network
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# Loss
loss = torch.nn.MSELoss()


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

# PDE损失
def l_interior(u):
    # 损失函数L1
    x, t, cond = interior()
    uxt = u(torch.cat([x, t], dim=1))
    uu = torch.exp(-(2*x-1)**2)*gradients(uxt, x, 1)
    return loss(gradients(uxt, t, 1) - 0.1*gradients(uu, x, 1), cond)

def l_left(u):
    # 损失函数L2
    x, t, cond = left()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt, x, 1) - 0.1 * (uxt - 1), cond)

def l_right(u):
    # 损失函数L3
    x, t, cond = right()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt, x, 1) + 0.1 * (uxt - 1), cond)

def l_down(u):
    # 损失函数L4
    x, t, cond = down()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(uxt, cond)

# Training

u = MLP()
opt = torch.optim.Adam(params=u.parameters(),lr=0.001)

lossvalues_x = []
lossvalues = []
for i in range(epochs):
    opt.zero_grad()
    l = l_interior(u) \
        + l_down(u) \
        + l_left(u) \
        + l_right(u)
    l.backward()
    opt.step()
    if (i+1) % 100 == 0:
        print("第%d代损失值为%.6f" % (i+1, l))
        lossvalues.append(l.detach().numpy())
        lossvalues_x.append(i+1)


# Inference
xx = torch.linspace(0, 1, h)
tt0 = torch.ones_like(xx) * 0
tt1 = torch.ones_like(xx) * 0.25
tt2 = torch.ones_like(xx) * 0.5
tt3 = torch.ones_like(xx) * 0.75
tt4 = torch.ones_like(xx) * 1
xx = xx.reshape(-1, 1)
tt0 = tt0.reshape(-1, 1)
tt1 = tt1.reshape(-1, 1)
tt2 = tt2.reshape(-1, 1)
tt3 = tt3.reshape(-1, 1)
tt4 = tt4.reshape(-1, 1)

u_pred0 = u(torch.cat([xx, tt0], dim=1))
u_pred1 = u(torch.cat([xx, tt1], dim=1))
u_pred2 = u(torch.cat([xx, tt2], dim=1))
u_pred3 = u(torch.cat([xx, tt3], dim=1))
u_pred4 = u(torch.cat([xx, tt4], dim=1))


###################  ploting  ##########################
# 1. 预测解图
plt.figure(figsize=(8, 5))
plt.grid()  # 生成网格
line1, = plt.plot(xx.detach().numpy(), u_pred0.detach().numpy(), color='red', linewidth=1, linestyle='-', label='t=0.0')
line2, = plt.plot(xx.detach().numpy(), u_pred1.detach().numpy(), color='b', linewidth=1, linestyle='-', label='t=0.25')
line3, = plt.plot(xx.detach().numpy(), u_pred2.detach().numpy(), color='k', linewidth=1, linestyle='-', label='t=0.5')
line4, = plt.plot(xx.detach().numpy(), u_pred3.detach().numpy(), color='y', linewidth=1, linestyle='-', label='t=0.75')
line5, = plt.plot(xx.detach().numpy(), u_pred4.detach().numpy(), color='gold', linewidth=1, linestyle='-', label='t=1.0')

plt.legend(handles=[line1, line2, line3, line4, line5], labels=['t=0.0', 't=0.25', 't=0.5', 't=0.75', 't=1.0'],loc='upper right')
plt.xlabel("x")
plt.ylabel("u")
plt.title("PINN")
plt.savefig('figure/pinn.png', dpi=600)
plt.show()

# 2. 损失值图
plt.figure(figsize=(8, 5))
plt.plot(lossvalues_x, lossvalues, color='red', linewidth=1, linestyle='-')
plt.xlabel("x")
plt.ylabel("loss")
plt.title("损失图")
plt.savefig('figure/loss.png', dpi=600)
plt.show()