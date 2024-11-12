import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

from agentPolicy import AgentLSTMPolicy, ProbeLSTMPolicy

from environment import AgentEnv, ProbeEnv

import gymnasium as gym

agent_env = AgentEnv()

agent_policy = AgentLSTMPolicy(agent_env.observation_space, agent_env.action_space, 3e-4, sensor_set=[])

agent_model = RecurrentPPO(agent_policy, agent_env)

agent_model.learn(5000)


probe_env = ProbeEnv(agent_policy)

probe_policy = ProbeLSTMPolicy(probe_env.observation_space, probe_env.action_space, 3e-4, sensor_set=[])

probe_model = RecurrentPPO(probe_policy, probe_env)

probe_model.learn(5000)


