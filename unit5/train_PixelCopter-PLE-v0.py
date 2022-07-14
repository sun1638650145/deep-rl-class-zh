"""使用Reinforce算法训练策略梯度."""
from collections import deque

import gym
import gym_pygame
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


def get_device():
    """获取可用的计算设备."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'当前可用的计算设备是: {device}')

    return device


class GradientPolicy(nn.Module):
    """策略梯度."""
    def __init__(self,
                 state_space,
                 action_space,
                 hidden_layer=16,
                 device=torch.device('cpu')):
        super(GradientPolicy, self).__init__()

        self.fc1 = nn.Linear(state_space, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, action_space)

        self.device = device  # 模型和张量使用的计算设备.

    def forward(self, x):
        """定义前向传播."""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.softmax(x, dim=-1)

    def action(self, state):
        """给定一个状态获得动作."""
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        probs = self.forward(state)

        # 创建一个样本空间进行采样.
        m = Categorical(probs.to('cpu'))  # 这里需要拷贝回cpu.
        action = m.sample()  # 采样到概率最大的动作可能性最大, 相比于argmax还可能进行探索.

        return action.item(), m.log_prob(action)


def train(env, policy, optimizer, n_training_episodes, max_steps, gamma, verbose=1):
    """训练模型."""
    rewards = deque(maxlen=100)  # 仅计算最近的100个样本.

    for episode in range(1, n_training_episodes + 1):
        state = env.reset()
        total_rewards_ep = []
        saved_log_probs = []

        for step in range(max_steps):
            # 使用策略梯度选择动作.
            action, log_prob = policy.action(state)

            # 采取行动.
            new_state, reward, done, info = env.step(action)

            total_rewards_ep.append(reward)
            saved_log_probs.append(log_prob)

            if done:
                break

            # 更新状态.
            state = new_state

        rewards.append(np.sum(total_rewards_ep))

        # 计算衰减系数\gamma^{k-t}
        discounts = [gamma ** i for i in range(len(total_rewards_ep))]  # 不要使用max_steps, 多数情况下已经提前退出了.
        # G_t=\sum^{T-1}_{k=t}\gamma^{k-t}r_k
        G_t = np.sum([discount * reward for discount, reward in zip(discounts, total_rewards_ep)])

        # 计算损失L(\theta)=\frac{1}{T}\sum^{T-1}_{t=0}G_t\log\pi_\theta(A_t|S_t).
        loss = []
        for log_prob in saved_log_probs:
            loss.append(-log_prob * G_t)  # 最小化-loss.
        loss = torch.cat(loss).sum()

        # 反向传播.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 显示训练进度信息.
        if verbose and episode % verbose == 0:
            print(f'第{episode}轮, 平均奖励: {np.mean(rewards):.2f}')


def evaluate(env, policy, n_eval_episodes, max_steps):
    """评估模型."""
    rewards = []

    for episode in range(n_eval_episodes):
        state = env.reset()
        total_rewards_ep = 0

        for step in range(max_steps):
            # 使用策略梯度选择动作.
            action, _ = policy.action(state)
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state

        rewards.append(total_rewards_ep)

    return np.mean(rewards), np.std(rewards)


if __name__ == '__main__':
    # 创建环境.
    env = gym.make(id='Pixelcopter-PLE-v0')

    # 获取设备.
    device = get_device()

    # 创建模型.
    policy = GradientPolicy(env.observation_space.shape[0],
                            env.action_space.n,
                            hidden_layer=64,
                            device=device).to(device)
    optimizer = optim.Adam(params=policy.parameters(),
                           lr=1e-4)

    # 训练模型.
    train(env=env,
          policy=policy,
          optimizer=optimizer,  # 使用的优化器.
          n_training_episodes=50000,  # 训练的总轮数.
          max_steps=1000,  # 每轮的最大步数.
          gamma=0.99,  # 衰减系数.
          verbose=1000)  # 是否显示日志.

    # 评估模型.
    mean_reward, std_reward = evaluate(env=env,
                                       policy=policy,
                                       n_eval_episodes=10,  # 测试的总轮数.
                                       max_steps=100)  # 每轮的最大步数.
    print(f'平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}')
