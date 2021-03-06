import metaworld
import random
import time

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks

env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

done = False
i = 0

start = time.time()
while not done:
    obs = env.reset()  # Reset environment
    a = env.action_space.sample()  # Sample an action
    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    i += 1
    env.render()
    if i % 100 == 0:
        print(i, time.time() - start)