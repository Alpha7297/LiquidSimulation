## 核函数

用B-spline核函数
$$
B_{2r}=\begin{cases}
\frac{3}{4}-r^2&|r|<1/2\\
\frac{1}{2}(\frac{3}{2}-|r|)^2&1/2\leq|r|\leq 3/2\\
0&|r|\geq 3/2
\end{cases}
$$
其中$r=\frac{x1-x2}{h}$，$h$是网格边长

## 理论建模

区别于传统方法，本项目采用拉格朗日描述与欧拉描述混合的仿真方法
网格分为empty solid和liquid三种类型
总粒子数为9倍liquid网格数
solid网格初始固定，之后不再修改
empty网格和liquid网格之间可以相互转换，当网格内存在粒子则为liquid网格，反之是empty
网格为交错网格，右侧边储存该网格x方向速度，下侧边储存该网格y方向速度
而网格中心储存压强信息

## 仿真迭代

1.进行P2G，将粒子速度投影到网格
2.加入外力，迭代更新压强，用更新后的压强修正速度
3.进行G2P，将网格速度投影到粒子
4.用速度更新位移

## 详细公式

### 1.P2G

首先用粒子坐标更新粒子所在的网格编号
之后更新网格类型，记录网格当前速度，之后清空
下一步对所有粒子，将自己的权重加入周围9个非solid格子
最后对于所有liquid格子，计算加权平均后的速度
$$
\vec{v_i}=\frac{\sum_j w_{ij} v_j}{\sum_j w_{ij}}
$$

### 2.Solve pressure

核心两个方程
$$
\rho (\frac{\partial v}{\partial t}+(v\cdot\nabla) v)=-\nabla p+\eta \nabla^2 v+f_{ext}\\
\nabla \cdot v=0
$$
边界条件
$$
p_{empty}=0\\
v_{solid}=0
$$
第一步，忽略压强获取新速度
$$
v_{new}=v+\frac{dt}{\rho}(f_{ext}+\eta \nabla^2 v)
$$
第二步，加入压强，求解压力泊松方程
要求$\nabla\cdot (v_new-\frac{dt}{\rho}\nabla p)=0$，即要求
$$
\nabla^2 p=\frac{\rho}{dt}\nabla\cdot v_{new}
$$
由于$v_new$并没有保证边界没有法向速度，因此这一步的边界条件是
$$
p_{air}=0\\
\frac{\partial p}{\partial n}|_{solid}=\frac{\rho}{dt}v_{new}\cdot \vec{n}
$$
之后使用Gauss-seidel迭代求解
第三步，用新的压强更新速度
$$
v_{next}=v_{new}-\frac{dt}{\rho}\nabla p
$$

### 3.G2P

这一步需要做的是将网格速度投影回粒子
方法使用PIC和FLIP混合

#### PIC

对于每个粒子，取周围九个网格更新该粒子的速度

$$
v_i=\frac{\sum_j w_{ij}v_j}{\sum_j w_{ij}}
$$

即得到PIC速度

#### FLIP

对于每个粒子，取周围9个网格更新该粒子的速度

$$
v_i=\frac{\sum_j w_{ij}(v_j^{curr}-v_j^{old})}{\sum_j w_{ij}}+v_i^{old}
$$

即得到FLIP速度

之后使用系数$\alpha$混合

$$
v_{new}=(1-\alpha)v_{PIC}+\alpha v_{FLIP}
$$

### 4.Update pos

用粒子的速度更新粒子的位移
之后渲染