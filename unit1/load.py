"""从Hugging Face Hub加载模型."""
import gym

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == '__main__':
    # 创建环境.
    eval_env = gym.make('LunarLander-v2')  # 验证环境.

    # 加载模型.
    model_file = load_from_hub(repo_id='sun1638650145/PPO-LunarLander-v2',
                               filename='PPO-LunarLander-v2.zip')
    model = PPO.load(model_file)

    # 评估模型.
    mean_reward, std_reward = evaluate_policy(model=model,
                                              env=eval_env,
                                              n_eval_episodes=10,
                                              deterministic=True)
    print(mean_reward, std_reward)
