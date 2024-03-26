import torch
import gym
import numpy as np
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn as nn

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCriticDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticDiscrete, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
                )

        # critic
        self.value_layer = nn.Sequential(
               nn.Linear(state_dim, 128),
               nn.ReLU(),
               nn.Linear(128, 64),
               nn.ReLU(),
               nn.Linear(64, 1)
               )

    # act 选择动作
    # evaluate 用于后续的更新策略，给更新策略提供需要的值

    # act单纯计算动作的概率，并没有涉及到log
    # evaluate需要更新，所以需要计算梯度，所以需要log

    def act(self, state, memory):
        # 将Numpy数组格式的当前状态state转换成PyTorch张量（Tensor）
        state = torch.from_numpy(state).float()

        # 计算当前状态下每个动作的概率
        action_probs = self.action_layer(state)

        # 根据动作概率创建一个分布
        dist = Categorical(action_probs)

        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        # 返回被抽样的动作的具体值，以便在环境中执行这个动作
        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)

        # 熵衡量了策略的随机性，高熵意味着策略更加随机
        # 添加熵可以防止策略过早收敛到次优动作，鼓励探索
        dist_entropy = dist.entropy()

        # state_value是对未来奖励的一个估计，反映了在给定状态下，根据当前策略π，期望获得的长期回报
        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, update_timestep):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.timestep = 0
        self.memory = Memory()
        self.update_timestep = update_timestep

        self.policy = ActorCriticDiscrete(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)  # betas是Adam中的参数
        self.policy_old = ActorCriticDiscrete(state_dim, action_dim)  # 初始化
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 计算函数的损失
        self.MseLoss = nn.MSELoss()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 归一化
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # torch.stack是用来组合一个张量列表的
        # self.memory.states收集了一系列的状态的对数概率
        # 通过使用torch.stack，可以将这些逐步收集的单个数据点组合成一个新的维度，即从列表形式的数据转换为批量的张量形式
        # 在后续梯度策略时候，需要使用到之前执行策略时的状态、动作和动作的对数概率来计算策略的优势函数和损失函数。
        # 这些计算往往依赖于整个批次的数据进行向量化操作，以提高效率和减少计算时间
        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()
        old_logprobs = torch.stack(self.memory.logprobs).detach()

        for _ in range(self.K_epochs):
            # 新策略重用旧样本进行训练
            # old_states, old_actions代表是旧的数据
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 比如我利用一组数据进行5次策略更新，那么5次策略更新使用的old_logprobs是一样的
            # 反正记住old_logprobs跟着action = agent.act(state)这句话改变
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算优势值
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages  # 重要性采样的思想，确保新的策略函数和旧策略函数的分布差异不大
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages  # 采样clip的方式过滤掉一些新旧策略相差较大的样本

            # 最小化损失函数

            # 最大化优势函数torch.min(surr1, surr2)=最小化‘-torch.min(surr1, surr2)’
            # 最大化熵=最小化‘- 0.01*dist_entropy’，鼓励熵增加

            # 最小化（状态值预测与实际奖励）之间的差异，实际上是指在更新值函数state_values的参数
            # 从而更准确地预测在策略π下从状态s开始的长期累积奖励
            # 优势函数并不一定因为值函数的准确预测而变为0，因为动作奖励具有随机性

            loss = -torch.min(surr1, surr2)  + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # 梯度下降优化
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 把当前策略的参数复制到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

    # 收集与环境交互的数据
    def step(self, reward, done):
        self.timestep += 1

        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

        if self.timestep % self.update_timestep == 0:
            self.update()
            self.memory.clear_memory()
            self.timestep = 0

    # 每一次更新完策略，old_logprobs需要根据policy_old重新计算
    # 在trainv2.py中action = agent.act(state)会调用这个，然后这里就会通过act更新old_logprobs
    # 所以old_logprobs并不是一个K_epochs中更新。
    def act(self, state):
        return self.policy_old.act(state, self.memory)