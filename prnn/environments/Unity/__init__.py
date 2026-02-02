import gymnasium as gym

gym.envs.registration.register(
    id="Unity-v0",
    entry_point="prnn.environments.Unity.UnityEnvironment:UnityEnv",
)
