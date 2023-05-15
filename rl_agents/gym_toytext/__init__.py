from gym.envs.registration import register

from rl_agents.gym_toytext.guessing_game import GuessingGame
from rl_agents.gym_toytext.hotter_colder import HotterColder
from rl_agents.gym_toytext.kellycoinflip import KellyCoinflipEnv, KellyCoinflipGeneralizedEnv
from rl_agents.gym_toytext.nchain import NChainEnv
from rl_agents.gym_toytext.riverswim import RiverSwimEnv
from rl_agents.gym_toytext.roulette import RouletteEnv
from rl_agents.gym_toytext.sixarms import SixArmsEnv


register(
    id="GuessingGame-v0",
    entry_point="gym_toytext:GuessingGame",
    max_episode_steps=200,
)

register(
    id="HotterColder-v0",
    entry_point="gym_toytext:HotterColder",
    max_episode_steps=200,
)

register(
    id="KellyCoinflip-v0",
    entry_point="gym_toytext:KellyCoinflipEnv",
    reward_threshold=246.61,
)

register(
    id="KellyCoinflipGeneralized-v0",
    entry_point="gym_toytext:KellyCoinflipGeneralizedEnv",
)

register(
    id="NChain-v0",
    entry_point="gym_toytext:NChainEnv",
    max_episode_steps=200,
)

register(
    id="RiverSwim-v0",
    entry_point="gym_toytext:RiverSwimEnv",
    max_episode_steps=200,
)

register(
    id="Roulette-v0",
    entry_point="gym_toytext:RouletteEnv",
    max_episode_steps=100,
)

register(
    id="SixArms-v0",
    entry_point="gym_toytext:SixArmsEnv",
    max_episode_steps=200
)