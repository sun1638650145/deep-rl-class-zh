"""使用PyTorch创建PPO模型并训练."""
import gym
import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam


def get_device():
    """获取可用的计算设备."""
    if torch.cuda.is_available():
        device_ = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device_ = torch.device('mps')
    else:
        device_ = torch.device('cpu')

    print(f'当前可用的计算设备是: {device_}')

    return device_


class PPO(nn.Module):
    """近端策略优化算法."""
    def __init__(self, env, device, n_steps=2048, batch_size=64):
        super(PPO, self).__init__()

        # 超参数.
        self.env = env
        self.device = device
        self.n_steps = n_steps
        self.batch_size = batch_size

        self.optimizer = None

        self.actor = nn.Sequential(
            self._layer_init(nn.Linear(np.asarray(self.env.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, self.env.single_action_space.n), std=0.01),
        )

        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(np.asarray(self.env.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 1), std=1),
        )

    @staticmethod
    def _layer_init(layer, std=np.sqrt(2), bias_const=0):
        """网络层初始化方式."""
        nn.init.orthogonal_(tensor=layer.weight, gain=std)  # 权重正交初始化.
        nn.init.constant_(tensor=layer.bias, val=bias_const)  # 偏置常量初始化.

        return layer

    def action_and_value(self, state):
        """给定一个状态获得动作和价值."""
        state = torch.from_numpy(state).to(self.device)
        probs = self.actor(state)

        # 创建一个样本空间进行采样.
        m = Categorical(logits=probs.to('cpu'))  # 这里需要拷贝回cpu.
        action = m.sample()  # 采样到概率最大的动作可能性最大, 相比于argmax还可能进行探索.

        return action, m.log_prob(action), m.entropy(), self.critic(state)

    def compile(self, optimizer):
        """编译模型."""
        self.optimizer = optimizer

    def fit(self, total_timesteps):
        """训练模型."""
        num_updates = total_timesteps // self.batch_size

        state = self.env.reset()

        for update in range(num_updates):
            for step in range(self.n_steps):
                # 使用actor产生动作, 使用critic评估价值.
                with torch.no_grad():
                    action, log_prob, _, value = self.action_and_value(state)


if __name__ == '__main__':
    # 创建环境.
    envs = gym.vector.make('LunarLander-v2',
                           num_envs=4,
                           asynchronous=False)

    # 获取设备.
    device = get_device()

    # 创建并编译模型.
    model = PPO(env=envs,
                device=device,
                n_steps=128).to(device)
    model.compile(optimizer=Adam(params=model.parameters(), lr=1e-4))

    # 训练模型.
    model.fit(total_timesteps=1000000)
