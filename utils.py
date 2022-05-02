import copy
import random

import numpy as np
import pandas as pd
import torch
import tqdm
import matplotlib.pyplot as plt
from rl_cookbook.envs.simulation import train_task_model

def reset_rand_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 2**32)
    print(f'Setting random seed to {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

def draw_colored_weights(weights, n_cols, n_rows, sz=(3, 40, 40), text=None, size=(10, 10)):
    weights = weights.detach().cpu().numpy()
    weights = weights.reshape((-1, *sz)).transpose(0, 2, 3, 1)
    # Hidden node indices
    indexes = np.random.randint(0, weights.shape[0], n_cols*n_rows)
    weights = weights[indexes]
    fig=plt.figure(figsize=size)
    HM = np.zeros((sz[1]*n_rows, sz[2]*n_cols, sz[0]))
    for idx in range(n_cols * n_rows):
        x, y = idx % n_cols, idx // n_cols
        HM[y*sz[1]:(y+1)*sz[1],x*sz[2]:(x+1)*sz[2]] = weights[idx]
    plt.clf()
    low, high = HM.min(), HM.max()
    plt.imshow((HM - low) / (high - low))
    if text is not None: plt.title(text)
    plt.axis('off')
    return HM

def draw_weights(weights, n_cols, n_rows, n_channels=3, sz=40, text=None, size=(10, 10)):
    weights = copy.deepcopy(weights).detach().cpu().numpy()
    weights = weights.reshape((-1, n_channels, sz, sz)).transpose(0, 2, 3, 1)
    weights = weights.mean(axis=3)
    indexes = np.random.randint(0, len(weights), n_cols*n_rows)
    weights = weights[indexes]
    fig=plt.figure(figsize=size)
    HM=np.zeros((sz*n_rows,sz*n_cols))
    for idx in range(n_cols * n_rows):
        x, y = idx % n_cols, idx // n_cols
        HM[y*sz:(y+1)*sz,x*sz:(x+1)*sz]=weights[idx]
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM, cmap='bwr', vmin=-nc, vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    if text is not None: plt.title(text)
    plt.axis('off')
    return weights

def run_benchmark(env, agent=None, create_agent=None, n_steps=int(1e5), n_runs=5):
    run_steps = []
    run_rewards = []
    for _ in tqdm(range(n_runs)):
        if agent:
            tmp_agent = copy.deepcopy(agent)
        else:
            tmp_agent = create_agent()
        steps, rewards = train_task_model(tmp_agent, env, n_steps, print_rewards=False)
        run_steps.append(steps)
        run_rewards.append(rewards)
    return run_steps, run_rewards

def moving_average(data, n=3):
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def bin_data(data, bin_size):
    binned_data = []
    for i, x in enumerate(data):
        if i % bin_size == 0:
            binned_data.append([])
        binned_data[-1].append(x)
    return binned_data

def generate_plot_data(x_data, y_data, bin_size=1):
    run_dfs = [pd.DataFrame({'x': x_series, 'y': y_series}) \
        for x_series, y_series in zip(x_data, y_data)]
    unique_xs = np.unique(np.concatenate([df['x'].values for df in run_dfs]))

    interp_dfs = []
    for i, run_df in enumerate(run_dfs):
        interp_data = []
        run_xs = set(run_df['x'])
        for x in unique_xs:
            if x not in run_xs:
                interp_data.append({
                    'x': x,
                    'y': np.nan,
                    'run': int(i)})
        interp_df = pd.DataFrame(interp_data)
        interp_df = pd.concat([interp_df, run_df])
        interp_df = interp_df.sort_values('x')
        interp_df['y'] = interp_df['y'].interpolate(limit_direction='forward')
        interp_dfs.append(interp_df)

    full_interp_df = pd.concat(interp_dfs).reset_index().drop('index', axis=1)
    full_interp_df.dropna(inplace=True)
    if bin_size > 1:
        x_bins = np.arange(full_interp_df['x'].min(), full_interp_df['x'].max() + bin_size, bin_size)
        bin_new_xs = (((np.arange(len(x_bins)-1) + 0.5) * bin_size)).astype(int) + full_interp_df['x'].min()
        full_interp_df['x_bin'] = pd.cut(full_interp_df['x'], bins=x_bins, labels=bin_new_xs, include_lowest=True)
        groups = full_interp_df.groupby(['run', 'x_bin'])['y'].mean()
        groups = groups.reset_index()
        groups.rename(columns={'x_bin': 'x'}, inplace=True)
        full_interp_df = groups
        
    return full_interp_df

# def generate_plot_data(y_data, bin_size=100, max_x=None):
#     if len(y_data.shape) == 2:
#         y_data = y_data.mean(axis=0)
#     binned_y = bin_data(y_data, bin_size)
#     binned_y = np.array([np.average(bin_data) for bin_data in binned_y])
#     if max_x is None:
#         max_x = len(binned_y)
#         x_data = np.arange(len(binned_y)) * bin_size
#     else:
#         x_data = np.arange(len(binned_y)) / len(binned_y) * max_x
#     print(binned_y)
#     return x_data, binned_y

def test_agent(agent, env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        global hist
        hist = []
        while not done:
            action = agent.sample_act(obs)
            obs, reward, done, _ = env.step(action)
            hist.append(obs)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)