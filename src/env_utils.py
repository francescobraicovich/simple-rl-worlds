# Contents for src/env_utils.py
import gymnasium as gym
import random

ATARI_GAMES = [
    "ALE/Adventure-v5",
    "ALE/AirRaid-v5",
    "ALE/Alien-v5",
    "ALE/Amidar-v5",
    "ALE/Assault-v5",
    "ALE/Asterix-v5",
    "ALE/Asteroids-v5",
    "ALE/Atlantis-v5",
    "ALE/BankHeist-v5",
    "ALE/BattleZone-v5",
    "ALE/BeamRider-v5",
    "ALE/Berzerk-v5",
    "ALE/Bowling-v5",
    "ALE/Boxing-v5",
    "ALE/Breakout-v5",
    "ALE/Carnival-v5",
    "ALE/Centipede-v5",
    "ALE/ChopperCommand-v5",
    "ALE/CrazyClimber-v5",
    "ALE/Crossbow-v5",
    "ALE/Defender-v5",
    "ALE/DemonAttack-v5",
    "ALE/DoubleDunk-v5",
    "ALE/ElevatorAction-v5",
    "ALE/Enduro-v5",
    "ALE/FishingDerby-v5",
    "ALE/Freeway-v5",
    "ALE/Frostbite-v5",
    "ALE/Galaxian-v5",
    "ALE/Gopher-v5",
    "ALE/Gravitar-v5",
    "ALE/Hero-v5",
    "ALE/IceHockey-v5",
    "ALE/Jamesbond-v5",
    "ALE/JourneyEscape-v5",
    "ALE/Kaboom-v5",
    "ALE/Kangaroo-v5",
    "ALE/KeystoneKapers-v5",
    "ALE/KingKong-v5",
    "ALE/Klax-v5",
    "ALE/Koolaid-v5",
    "ALE/Krull-v5",
    "ALE/KungFuMaster-v5",
    "ALE/MarioBros-v5",
    "ALE/MontezumaRevenge-v5",
    "ALE/MsPacman-v5",
    "ALE/NameThisGame-v5",
    "ALE/Phoenix-v5",
    "ALE/Pitfall-v5",
    "ALE/Pong-v5",
    "ALE/Pooyan-v5",
    "ALE/PrivateEye-v5",
    "ALE/Qbert-v5",
    "ALE/Riverraid-v5",
    "ALE/RoadRunner-v5",
    "ALE/Robotank-v5",
    "ALE/Seaquest-v5",
    "ALE/Skiing-v5",
    "ALE/Solaris-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/StarGunner-v5",
    "ALE/Tennis-v5",
    "ALE/TimePilot-v5",
    "ALE/Tutankham-v5",
    "ALE/UpNDown-v5",
    "ALE/Venture-v5",
    "ALE/VideoPinball-v5",
    "ALE/WizardOfWor-v5",
    "ALE/YarsRevenge-v5",
    "ALE/Zaxxon-v5",
]

def get_env_details(env_name):
    if env_name == "atari":
        actual_env_name = random.choice(ATARI_GAMES)
    else:
        actual_env_name = env_name

    temp_env = gym.make(actual_env_name)
    action_space = temp_env.action_space
    observation_space = temp_env.observation_space

    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
        action_type = 'discrete'
    elif isinstance(action_space, gym.spaces.Box):
        action_dim = action_space.shape[0]
        action_type = 'continuous'
    else:
        temp_env.close()
        raise ValueError(
            f"Unsupported action space type: {type(action_space)}")

    temp_env.close()
    print(f"Environment: {actual_env_name}")
    print(f"Action space type: {action_type}, Action dimension: {action_dim}")
    print("Raw observation space:", observation_space)
    return action_dim, action_type, observation_space
