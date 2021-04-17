import gym
env = gym.make('MsPacman-v0')

env.seed(42)
obs = env.reset()

obs, reward, done, info = env.step(0)

frames = []

n_max_steps = 1000
n_change_steps = 10

env.seed(42)
obs = env.reset()
for step in range(n_max_steps):
    img = env.render(mode="rgb_array")
    frames.append(img)
    if step % n_change_steps == 0:
        action = env.action_space.sample() # play randomly
    obs, reward, done, info = env.