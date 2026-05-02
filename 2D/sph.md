## SPH项目理论基础

### 1. 状态方程

密度由 SPH 核函数加权求和得到：

$$
\rho_i=m_iW(0)+\sum_jm_jW(r_{ij})
$$

其中 `j` 遍历距离小于 `2h` 的邻居粒子。初始化后会用所有流体粒子的平均密度作为静止密度：

$$
\rho_0=\frac{1}{N}\sum_i\rho_i
$$

压力使用弱可压缩 SPH 的状态方程：

$$
p_i=\max\left(0,\frac{k}{\gamma}\left[\left(\frac{\rho_i}{\rho_0}\right)^\gamma-1\right]\right)
$$

### 2. 运动方程和更新公式

流体运动方程为：

$$
\rho_i\frac{\mathrm{D}v_i}{\mathrm{D}t}
=-\nabla p+\eta\nabla^2v_i+f_{ext}
$$

其中右侧三项分别是压力项、粘性项和外力项。离散时先计算密度：

$$
\rho_i=\sum_jm_jW_{ij}(r)
$$

弱可压缩 SPH 中不再求解压力修正的泊松方程，而是由状态方程直接得到压力，然后把压力项、粘性项和外力项合并到加速度中：

$$
a_i=
-\sum_jm_j
\left(
\frac{p_i}{\rho_i^2}
+\frac{p_j}{\rho_j^2}
\right)
\nabla W_{ij}
+\eta\sum_jm_j\frac{v_j-v_i}{\rho_j}W_{ij}
-\sum_jm_j\Pi_{ij}\nabla W_{ij}
+f_{ext}
$$

其中压力梯度使用对称形式，保证动量守恒。普通粘性项使用速度差做扩散，人工粘性项 $\Pi_{ij}$ 用来耗散局部压缩速度。两种粘性项都只对真实流体粒子计算，不让虚粒子参与粘性。

为了避免粒子在碰撞或局部压缩时距离迅速靠近，WCSPH 代码中额外加入了人工粘性项。它只在两个粒子互相接近时生效：

$$
\Pi_{ij}=
\begin{cases}
\frac{-\alpha c\mu_{ij}}{\bar{\rho}_{ij}},&v_{ij}\cdot r_{ij}<0\\
0,&v_{ij}\cdot r_{ij}\ge0
\end{cases}
$$

其中：

$$
\mu_{ij}=\frac{h(v_{ij}\cdot r_{ij})}{r_{ij}^2+\epsilon h^2}
$$

这个项的目的不是增加普通粘性，而是耗散接近速度，抑制压强尖峰。尖峰出现的具体原因是边界处把粒子压回边界内部，可能导致和其他粒子迅速接近。

而不可压缩 SPH 中不使用状态方程直接给出压力，而是先忽略压力项，计算预测速度：

$$
v_i^*=v_i^n+\Delta t
\left(
\eta\sum_jm_j\frac{v_j-v_i}{\rho_0}W_{ij}
+f_{ext}
\right)
$$

然后要求修正后的速度满足不可压缩条件：

$$
\nabla\cdot v^{n+1}=0
$$

压力修正写成：

$$
v_i^{n+1}=v_i^*
-\Delta t\sum_jm_j
\left(
\frac{p_i+p_j}{\rho_0^2}
\right)\nabla W_{ij}
$$

因此需要求解压力场的泊松方程。代码中先用预测速度计算离散散度：

$$
(\nabla\cdot v^*)_i
=\sum_j\frac{m_j}{\rho_0}(v_j^*-v_i^*)\cdot\nabla W_{ij}
$$

泊松方程右端为：

$$
b_i=-\frac{\rho_0}{\Delta t}(\nabla\cdot v^*)_i
$$

为了在粒子上求解这个方程，代码把拉普拉斯项写成邻居粒子的加权形式：

$$
A_{ij}=
\frac{2m_j}{\rho_0}
\frac{-r_{ij}\cdot\nabla W_{ij}}{r_{ij}^2+\epsilon h^2}
$$

然后使用 Jacobi 迭代压力：

$$
p_i^{k+1}
=\max\left(
0,
\frac{b_i+\sum_jA_{ij}p_j^k}{\sum_jA_{ij}}
\right)
$$

这里压力不允许为负，因为负压会把粒子互相吸在一起，容易造成非物理聚团。虚粒子在散度和对角项中提供不可穿透边界，但压力迭代时只使用真实流体粒子的压力；迭代结束后再由附近流体压力外推虚粒子压力，用于速度修正。

之后更新速度和位置。对于弱可压缩 SPH，速度由加速度显式更新：

$$
v_i^{n+1}=v_i^n+\Delta t\,a_i
$$

对于不可压缩 SPH，速度已经在压力修正步骤中得到：

$$
v_i^{n+1}=v_i^*
-\Delta t\sum_jm_j
\left(
\frac{p_i+p_j}{\rho_0^2}
\right)\nabla W_{ij}
$$

两种方法最后都用新的速度更新位置：

$$
x_i^{n+1}=x_i^n+\Delta t\,v_i^{n+1}
$$

### 3. 样条核函数

代码使用二维 cubic spline kernel。令：

$$
q=\frac{r}{h}
$$

则核函数为：

$$
W(r,h)=\frac{10}{7\pi h^2}
\begin{cases}
1-\frac{3}{2}q^2+\frac{3}{4}q^3,&0\le q\le1\\
\frac{1}{4}(2-q)^3,&1<q\le2\\
0,&q>2
\end{cases}
$$

它的导数在代码中写成 `dW(r)`：

$$
\frac{\partial W}{\partial r}
=\frac{10}{7\pi h^3}
\begin{cases}
-3q+\frac{9}{4}q^2,&0\le q\le1\\
-\frac{3}{4}(2-q)^2,&1<q\le2\\
0,&q>2
\end{cases}
$$

邻居搜索只需要考虑 `r<2h` 的粒子。

### 4. 边界处理

边界由两部分组成。

第一部分是三层虚粒子。底部和左右侧壁都布置了固定虚粒子。

它们参与密度计算，用来补足墙附近缺失的邻居，防止流体粒子在边界处密度偏低。虚粒子也会按照状态方程计算压力，并参与流体粒子的压力梯度项，因此主要边界作用来自虚粒子的压力排斥。

虚粒子不参与普通粘性项和人工粘性项，避免固定边界粒子把流体速度直接粘住。

第二部分是几何约束。粒子位置更新后，如果数值上越过容器边界，就把位置放回边界，并把朝墙外的法向速度置零；正常情况下应主要依赖虚粒子压力防止穿透。
