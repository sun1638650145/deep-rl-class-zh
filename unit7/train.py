"""创建环境并训练A2C模型."""
import gym
import pybullet_envs  # 用于创建第三方环境.

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == '__main__':
    # 创建环境.
    env = gym.make('AntBulletEnv-v0')

    # 创建一组并行环境.
    envs = make_vec_env('AntBulletEnv-v0', n_envs=4)
    # 创建模型并训练.
    model = A2C(policy='MlpPolicy',
                env=envs,
                learning_rate=0.0096,
                n_steps=8,
                gamma=0.99,
                gae_lambda=0.9,
                ent_coef=0.0,
                vf_coef=0.4,
                max_grad_norm=0.5,
                use_rms_prop=True,
                use_sde=True,
                normalize_advantage=False,
                tensorboard_log='./tensorboard',
                policy_kwargs=dict(log_std_init=2,
                                   ortho_init=False),
                verbose=1)
    model.learn(total_timesteps=2000000)
    # 保存模型.
    # model.save('./a2c-AntBulletEnv-v0')

    # 禁止更新参数和标准化.
    env.training = False
    env.norm_reward = False
    # 评估模型并返回平均奖励.
    mean_reward, std_reward = evaluate_policy(model=model,
                                              env=env)
    print(mean_reward, std_reward)
