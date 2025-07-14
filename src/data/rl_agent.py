# src/rl_agent.py
 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

def create_ppo_agent(vec_env: VecEnv, ppo_config: dict, device: str = 'mps', load_agent: bool = False):
    """
    Create a PPO agent with the given configuration and environment.
    """
    save_path = 'best_models/ppo_agent'
    if load_agent:
        agent = load_ppo_agent(save_path, vec_env)
        return agent

    agent = PPO(
        ppo_config.get('policy_type', 'MlpPolicy'),
        vec_env,
        learning_rate=ppo_config.get('learning_rate'),
        n_steps=ppo_config.get('n_steps'),
        batch_size=ppo_config.get('batch_size'),
        n_epochs=ppo_config.get('n_epochs'),
        gamma=ppo_config.get('gamma'),
        gae_lambda=ppo_config.get('gae_lambda'),
        clip_range=ppo_config.get('clip_range'),
        ent_coef=ppo_config.get('ent_coef', 0.01),  # Add entropy coefficient for exploration
        tensorboard_log=None,
        verbose=1,
        device=device,
    )
    return agent

def train_ppo_agent(agent: PPO, ppo_config: dict, task_name: str = "PPO Training"):
    """
    Train the PPO agent with the specified configuration.
    """
    total_timesteps = ppo_config.get('total_train_timesteps', 100000)
    print(f"Starting {task_name} for {total_timesteps} timesteps...")
    agent.learn(total_timesteps=total_timesteps, progress_bar=True)
    print(f"{task_name} complete.")


def save_ppo_agent(agent: PPO, save_path: str):
    """
    Save the trained PPO agent to the specified path.
    """
    agent.save(save_path)
    print(f"Agent saved to {save_path}")

def load_ppo_agent(save_path: str, vec_env: VecEnv):
    """
    Load a PPO agent from the specified path.
    """
    agent = PPO.load(save_path, env=vec_env)
    print(f"Agent loaded from {save_path}")
    return agent