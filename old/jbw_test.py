from math import pi

import matplotlib.pyplot as plt
import numpy as np
import jbw

items = []
items.append(jbw.Item("banana", [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0], [0], False, 0.0,
        intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[-5],
        interaction_fns=[[jbw.InteractionFunction.ZERO]]))

# construct the simulator configuration
config = jbw.SimulatorConfig(max_steps_per_movement=1, vision_range=1,
  allowed_movement_directions=[jbw.ActionPolicy.ALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED],
  allowed_turn_directions=[jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.ALLOWED, jbw.ActionPolicy.ALLOWED],
  no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items, agent_color=[0.0, 0.0, 1.0],
  collision_policy=jbw.MovementConflictPolicy.FIRST_COME_FIRST_SERVED, agent_field_of_view=2*pi,
  decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000)

def reward_fn(prev_items, items):
    """
    Reference for item indicies:
    0 - Banana: 0 reward
    1 - Onion: -1 reward for every one collected
    2 - JellyBean: +1 reward for every one collected
    3 - Wall: 0 reward, cannot collect
    4 - Tree: 0 reward, cannot collect
    5 - Truffle: 0 reward
    """
    reward_array = np.array([0, -1, 1, 0, 0, 0])
    diff = items - prev_items

    return (diff * reward_array).sum().astype(np.float32)

env = jbw.JBWEnv(
    sim_config = config,
    reward_fn = reward_fn,
    render = True)

env.render()
input()