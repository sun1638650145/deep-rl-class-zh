"""从Hugging Face Hub加载模型."""
import gym
import numpy as np
import pickle5 as pickle
from huggingface_hub import hf_hub_download


def greedy_policy(Qtable, state):
    """greedy策略."""

    # 经验: 采取具有最大值的动作.
    action = np.argmax(Qtable[state])

    return action


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


def load_from_hub(repo_id: str,
                  filename: str) -> dict:
    """从Hugging Face Hub下载模型.

    Args:
        repo_id: Hugging Face Hub中模型仓库id.
        filename: 仓库中的zip文件名.

    Return:
        从Hugging Face Hub下载的模型.
    """
    pickle_model = hf_hub_download(repo_id=repo_id,
                                   filename=filename,
                                   cache_dir='./')

    with open(pickle_model, 'rb') as fp:
        downloaded_model_file = pickle.load(fp)

    return downloaded_model_file


if __name__ == '__main__':
    # 加载模型.
    model = load_from_hub(repo_id='sun1638650145/q-FrozenLake-v1-4x4-noSlippery',
                          filename='q-learning.pkl')

    # 创建环境.
    env = gym.make(model['env_id'], is_slippery=False)
    env.reset()

    # 评估模型.
    mean_reward, std_reward = evaluate(env=env,
                                       max_steps=model['max_steps'],
                                       n_eval_episodes=model['n_eval_episodes'],
                                       Qtable=model['qtable'])
    print(mean_reward, std_reward)
