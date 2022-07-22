"""创建环境并训练PPO模型."""
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def show_env(env):
    """查看环境."""
    # 重置环境.
    env.reset()

    for _ in range(200):
        # 渲染环境.
        env.render()
        # 获取随机动作.
        action = env.action_space.sample()
        # 执行动作.
        env.step(action)

    # 关闭环境.
    env.close()


if __name__ == '__main__':
    # 创建环境.
    env = gym.make('LunarLander-v2')
    # show_env(env)

    # 创建一组并行环境.
    envs = make_vec_env('LunarLander-v2', n_envs=16)
    # 创建模型并训练.
    model = PPO(policy='MlpPolicy',
                env=envs,
                n_steps=128,
                batch_size=128,
                n_epochs=4,
                gamma=0.999,
                gae_lambda=0.98,
                ent_coef=0.01,
                verbose=1)
    model.learn(total_timesteps=1000000)
    # 保存模型.
    # model.save('./ppo-LunarLander-v2')

    # 评估模型并返回平均奖励.
    mean_reward, std_reward = evaluate_policy(model=model,  # base_class.BaseAlgorithm|你想评估的模型.
                                              env=env,  # gym.env|Gym环境.
                                              n_eval_episodes=10,  # int|10|评估周期.
                                              deterministic=True)  # bool|True|使用确定动作还是随机动作.
    print(mean_reward, std_reward)
