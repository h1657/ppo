# 实现ppo三种变种算法
## 实验报告
### 一、实验内容
1.完成PPO三种变种算法  
2.实现在LunarLander-v2环境下的训练
### 二、实验过程
#### 1.配置环境
##### (1)通过cmd配置虚拟环境gym  
```python
conda create -n gym python=3.8.3
```
##### (2)安装pytorch框架和gym的一般库
  ```python
conda install pytorch
pip3 install gym matplotlib -i  https://pypi.tuna.tsinghua.edu.cn/simple
  ```
##### (3)安装box2d
```python
conda install -c conda-forge box2d-py
```
##### (4)安装pygame  
通过下载pygame-2.1.2-cp38-cp38-win_amd64.whl进行安装，在Unofficial Windows Binaries for Python Extension Packages网站下载好pygame，并将文件放到了Python的安装目录下（并新建文件夹 mypackage），再在虚拟环境进行下载
```python
pip install "D:\Python3.8位置\mypackage\pygame-2.1.2-cp38-cp38-win_amd64.whl"
```
##### (5)测试环境  
运行以下代码没有报错证明环境配置成功
  ```python
import gym
env = gym.make('LunarLander-v2' ,render_mode='rgb_array')
  ```

