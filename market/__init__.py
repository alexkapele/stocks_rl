from gym.envs.registration import register

register(
    id='market-v0',
    entry_point='gym_market.envs:MarketEnv',
)
