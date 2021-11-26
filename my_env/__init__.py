from gym.envs.registration import register


register(
    id='foraging-v0',
    entry_point='my_env.envs:foraging',
)