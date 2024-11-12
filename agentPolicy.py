from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
import torch

class AgentLSTMPolicy(RecurrentMultiInputActorCriticPolicy):

    def __init__(self, *args, sensor_set, **kwargs):

        super().__init__(*args, **kwargs)
        self.sensor_set = sensor_set


    def forward(self, obs, lstm_states, episode_starts, deterministic):
        masked_obs = {}
        for key, value in obs.items():
            if key in self.sensor_set:
                masked_obs[key] = value

            else:
                masked_obs[key] = torch.randn_like(value)



        return super().forward(masked_obs, lstm_states, episode_starts, deterministic)



class ProbeLSTMPolicy(RecurrentMultiInputActorCriticPolicy):

    def __init__(self, *args, sensor_set, **kwargs):
        super().__init_(*args, **kwargs)
        self.sensor_set = sensor_set


    def forward(self, obs, lstm_states, episode_starts, deterministic):
        if episode_starts:
            lstm_states = self.env.get_final_state()

        masked_obs = {}
        for key, value in obs.items():
            if key in self.sensor_set:
                masked_obs[key] = value

            else:
                masked_obs[key] = torch.randn_like(value)

        return super().forward(masked_obs, lstm_states, episode_starts, deterministic)
