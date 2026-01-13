from gymnasium.envs.registration import register

register(
    id="SnakeWorld-v0",
    entry_point="gymnasium_env.envs.snake_game:SnakeWorldEnv",
    max_episode_steps=2000, # Optional: Limits game to 2000 steps so it doesn't run forever
)