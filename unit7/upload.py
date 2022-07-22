"""发布模型到Hugging Face Hub."""
import gym
import pybullet_envs

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_sb3 import package_to_hub

model = A2C.load('a2c-AntBulletEnv-v0', print_system_info=True)
model_name = 'a2c-AntBulletEnv-v0'
model_architecture = 'A2C'
env_id = 'AntBulletEnv-v0'
eval_env = DummyVecEnv([lambda: gym.make(env_id)])
repo_id = 'sun1638650145/A2C-AntBulletEnv-v0'
commit_message = 'AntBulletEnv-v0 A2C 1st'


package_to_hub(model=model,
               model_name=model_name,
               model_architecture=model_architecture,
               env_id=env_id,
               eval_env=eval_env,
               repo_id=repo_id,
               commit_message=commit_message)
