"""发布模型到Hugging Face Hub."""
import gym

from huggingface_sb3 import package_to_hub
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


model = PPO.load('PPO-LunarLander-v2', print_system_info=True)
model_name = 'PPO-LunarLander-v2'
model_architecture = 'PPO'
env_id = 'LunarLander-v2'
eval_env = DummyVecEnv([lambda: gym.make(env_id)])
repo_id = 'sun1638650145/PPO-LunarLander-v2'
commit_message = 'LunarLander-v2 uses the PP0 algorithm.'

package_to_hub(model=model,
               model_name=model_name,
               model_architecture=model_architecture,
               env_id=env_id,
               eval_env=eval_env,
               repo_id=repo_id,
               commit_message=commit_message)
