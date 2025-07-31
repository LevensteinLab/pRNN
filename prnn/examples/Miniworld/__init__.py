import gymnasium as gym

# Register the environment with OpenAI Gym
gym.envs.registration.register(
    id="MiniWorld-LRoom-v0",
    entry_point="prnn.examples.Miniworld.env:LRoom",
    # kwargs={'continuous': True},
)

gym.register(
    id="MiniWorld-LRoom-v1",
    entry_point="prnn.examples.Miniworld.env:LRoom",
    kwargs={"walls": ("brick_wall", "marble", "wood_planks"),
            "floors": ("asphalt", "floor_tiles_bw", "concrete_tiles"),
            "sheep": True},
)

