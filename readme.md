# 项目须知

## 版本配置

taichi 1.7.4
python 3.11.15

## 项目内容

包含2D和3D两个文件夹，每个文件夹中两个子项目，PIC/FLIP和WCSPH

这三个项目初始条件均相同，目的是为了对比不同的仿真方法对实际结果的影响

最终分析与对比在analyse.md中

PIC_FLIP.md介绍了PIC/FLIP的理论基础

sph.md介绍了sph仿真的理论基础

IDP_solution.md中我尝试自行提出了一种修正流体密度不守恒的方法

视频在videos中，由于每帧都要保存，3D速度及其缓慢，因此只有2D的视频