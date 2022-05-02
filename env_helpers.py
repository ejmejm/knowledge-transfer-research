import numpy as np
import gym
import cv2
from gym_minigrid.envs import EmptyEnv
from gym_minigrid.minigrid import Grid
from gym_minigrid.wrappers import *

class GridEnv(EmptyEnv):
    def __init__(
            self,
            size=8,
            agent_start_pos=None,
            agent_start_dir=0,
            goal_start_pos=None):
        self.goal_start_pos = goal_start_pos
        super().__init__(size, agent_start_pos, agent_start_dir)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        if self.goal_start_pos is None:
            self.put_obj(Goal(), self._rand_int(1, width-1), self._rand_int(1, width-1))
        else:
            self.put_obj(Goal(), self.goal_start_pos[0], self.goal_start_pos[1])

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

class SimplifyObsWrapper(gym.ObservationWrapper):
  def __init__(self, env, grayscale=True, scale_factor=-1, rescale=True):
    super().__init__(env)
    self.grayscale = grayscale
    self.scale_factor = scale_factor
    self.rescale = rescale
    obs_shape = self.env.observation_space.shape
    if scale_factor > 0:
      new_shape = [int(obs_shape[0] * scale_factor), int(obs_shape[1] * scale_factor)]
    else:
      new_shape = [obs_shape[0], obs_shape[1]]
    if not grayscale:
      new_shape = [3] + new_shape

    self.observation_space = gym.spaces.Box(
        low=0, high=1, shape=new_shape, dtype=np.float32)

  def observation(self, observation):
    # Downscale by a factor of 2
    if self.scale_factor > 0 and self.scale_factor != 1:
      observation = cv2.resize(observation, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

    # Convert to grayscale
    if self.grayscale:
      observation = np.sum(observation * np.array([[[0.2989, 0.5870, 0.1140]]]), axis=2)
    else:
      observation = observation.transpose(2, 0, 1)

    # Rescale obs to [0, 1]
    if self.rescale:
      observation = observation / 255.0
      
    return observation

class ColorShiftWrapper(gym.ObservationWrapper):
  def __init__(self, env, color=None, strength=0.3):
    super().__init__(env)
    if color is None:
      self.color = (np.random.rand(), np.random.rand(), np.random.rand())
    else:
      self.color = color
    self.color = np.array(self.color) * 255
    self.strength = strength
    self.filter_img = np.full(
      (self.env.observation_space.shape[0], self.env.observation_space.shape[1], 3),
      (self.color), dtype=np.float32)

  def observation(self, observation):
    return observation * (1 - self.strength) + self.filter_img * self.strength

def color_shift(obs, color, strength=0.3):
  color_arr = np.array(color).reshape(1, 1, 3)
  return obs * (1 - strength) + color_arr * strength

def reverse_color_shift(obs, color, strength=0.3):
  color_arr = np.array(color).reshape(1, 1, 3)
  return (obs - color_arr * strength) / (1 - strength)

class FlattenObservations(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)
    obs_shape = self.env.observation_space.shape
    self.observation_space = gym.spaces.Box(
        low=0, high=1, shape=(np.prod(obs_shape),), dtype=np.float32)

  def observation(self, observation):
    return observation.reshape(-1)

class SimplifyActionsWrapper(gym.ActionWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.action_space = gym.spaces.Discrete(3)

  def action(self, action):
    if action > 2:
      raise ValueError('Action must be between 0 and 2')
    return action