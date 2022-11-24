"""训练Q-Learning智能体."""
import gym
import numpy as np


def initialize_q_table(state_space, action_space):
    """初始化Q-表."""
    Qtable = np.zeros([state_space, action_space])

    return Qtable


def epsilon_greedy_policy(Qtable, state, epsilon):
    """epsilon-greedy策略."""
    # 随机生成一个在0和1之前的数字.
    random_num = np.random.uniform(0, 1)
    # 使用经验.
    if random_num > epsilon:
        # 从给定状态中采取最大值的动作.
        action = np.argmax(Qtable[state])
    # 进行探索.
    else:
        action = env.action_space.sample()  # 进行随机动作.

    return action


def greedy_policy(Qtable, state):
    """greedy策略."""
    # 经验: 采取具有最大值的动作.
    action = np.argmax(Qtable[state])

    return action


def train(env, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, Qtable, learning_rate, gamma):
    """训练模型."""
    for episode in range(n_training_episodes):
        # 减小epsilon(因为我们需要越来越少的探索).
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        # 重置环境.
        state = env.reset()

        # 循环.
        for step in range(max_steps):
            # 使用epsilon-greedy策略选择动作At.
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            # 采取动作At并观察Rt+1和St+1,
            # 采取动作(a)并观察结果状态(s')和奖罚(r).
            new_state, reward, done, info = env.step(action)

            # 更新Q(s, a) = Q(s, a) + lr[R(s, a) + gamma * max Q(s', a') - Q(s, a)]
            Qtable[state][action] += (
                    learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]))

            if done:
                break

            # 更新状态.
            state = new_state

    return Qtable


def evaluate(env, max_steps, n_eval_episodes, Qtable):
    """评估模型."""
    rewards = []

    for episode in range(n_eval_episodes):
        state = env.reset()
        total_rewards_ep = 0

        for step in range(max_steps):
            # 在给定状态下, 采取具有最大期望奖励的动作(索引).
            action = greedy_policy(Qtable, state)
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state

        rewards.append(total_rewards_ep)

    return np.mean(rewards), np.std(rewards)


if __name__ == '__main__':
    # 创建环境.
    env = gym.make(id='FrozenLake-v1',
                   map_name='4x4',
                   is_slippery=False)
    env.reset()

    # 初始化Q-表.
    Qtable_frozenlake = initialize_q_table(state_space=env.observation_space.n,
                                           action_space=env.action_space.n)
    # 训练模型.
    Qtable_frozenlake = train(env=env,
                              n_training_episodes=10000,  # 训练的总轮数.
                              min_epsilon=0.05,  # 最小探索率.
                              max_epsilon=1.0,  # 初始探索率.
                              decay_rate=0.0005,  # 指数探索衰减率.
                              max_steps=99,  # 每轮的最大步数.
                              Qtable=Qtable_frozenlake,
                              learning_rate=0.7,  # 学习率.
                              gamma=0.95)  # 衰减系数.

    # 评估模型并返回平均奖励.
    mean_reward, std_reward = evaluate(env=env,
                                       max_steps=99,  # 每轮的最大步数.
                                       n_eval_episodes=100,  # 测试的总轮数.
                                       Qtable=Qtable_frozenlake)
    print(mean_reward, std_reward)
