---
categories:
- ai
- huggingface
- reinforcement-learning
- notes
date: 2022-5-26
description: Unit 2 introduces monte carlo, temporal difference learning, and Q-learning.
image: ./images/frozen_lake.gif
hide: false
layout: post
search_exclude: false
title: Notes on The Hugging Face Deep RL Class Pt.2
toc: false

aliases:
- /Notes-on-HF-Deep-Reinforcement-Learning-Class-2/
---

* [Types of Value-Based Methods](#types-of-value-based-methods)
* [The Bellman Equation](#the-bellman-equation)
* [Monte Carlo vs Temporal Difference Learning](#monte-carlo-vs-temporal-difference-learning)
* [Introducing Q-Learning](#introducing-q-learning)
* [Lab](#lab)
* [References](#references)


## Types of Value-Based Methods

- The value of a state is the expected discounted return from starting in that state and following the policy.
- Value-based methods involve learning a value function that maps a state to the expected value of being in that state.
- Finding an optimal value function leads to having an optimal policy.
- $\pi^{\*}(s) = argmax_{a} Q^{\*}(s,a)$
- Value-based methods require us to define how the agent acts (i.e., the policy) based on the predicted value map.
- Greedy policies always take the action that leads to the biggest reward.
- Epsilon-Greedy policies switch between exploring random actions and taking actions with the highest known reward.
    - The probability of exploring random actions is high at the beginning of training and decreases as training progresses.

### The State-Value function

- The state-value function, for each state $S_{t}$, outputs the expected return $E_{\pi}\left[ G_{t} \right]$ if the agent starts in that state $S_{t}$ and then follows the policy $\pi$ forever.
- $$V_{\pi}(s) = E_{\pi}\left[ G _{t} \vert S_{t} = s \right]$$

### The Action-Value function

- The action-value function outputs the expected return $E_{\pi}\left[ G_{t} \right] $ for each state-action pair $\left( S_{t}, A_{t} \right)$ if the agent takes a given action $A_{t}$ when starting in a given state $S_{t}$ and then follows the policy $\pi$ forever.
- $$Q_{\pi} (s,a) = E_{\pi} \left[ G_{t} \vert S_{t} = s, A_{t} = a \right]$$


## The Bellman Equation

- The Bellman equation simplifies our value estimation.
- The Bellman equation is a recursive equation that allows us to consider the value of any state $S_{t}$ as the immediate reward $R_{t+1}$ plus the discounted value of the state that follows $gamma \cdot V(S_{t+1})$.
- $V_{\pi}(s) = E_{\pi} \left[ R_{t+1}  + \gamma \cdot V_{\pi}(S_{t+1}) \vert S_{t} = s \right]$



## Monte Carlo vs Temporal Difference Learning

- Monte Carlo uses an entire episode of experience before learning.
- Temporal difference learning learns after each step.

### Monte Carlo: learning at the end of the episode

- Monte Carlo waits until the end of the episode, calculates the total rewards $G_{t}$, and uses it as a target for updating the value function $V(S_{t})$ using a learning rate $\alpha$.
- $V(S_{t}) \leftarrow V(S_{t}) + \alpha \left[G_{t} - V(S_{t}) \right]$
- At the end of each episode, we have a list of States, Actions, Rewards, and new States.
- The agent improves by running more and more episodes.
- Monte Carlo uses the actual accurate discounted return of an episode.

### Temporal Difference Learning: learning at each step

- Temporal difference waits for one interaction $S_{t+1}$, forms a TD target $R_{t+1} + \gamma \cdot V(S_{t+1})$, and updates the value function $V(S_{t})$ using the immediate reward plus $R_{t+1}$ the discounted value of the following state $gamma \cdot V(S_{t+1})$ scaled by a learning rate $\alpha$.
- $V(S_{t}) \leftarrow V(S_{t}) + \alpha \left[R_{t+1} + \gamma \cdot V(S_{t+1}) - V(S_{t}) \right]$
- TD Learning that waits for one step is TD(0) or one-step TD.
- The agent improves by running more and more steps.
- TD Learning uses an estimated return called TD target.


## Introducing Q-Learning

### What is Q-Learning?

- Q-Learning (a.k.a. Sarsamax) is an off-policy value-based method that uses a TD approach to train its action-value function called the Q-Function.
- Off-policy refers to using a different policy for acting and updating.
    - We use a greedy policy for updating the action-value function and an epsilon-greedy function for choosing actions.
- The "Q" refers to the quality of a given action in a given state.
- The Q-Function maintains a Q-table that tracks the value of each possible state-action pair.
- Each cell in the Q-table stores the value from taking a given action in a given state.
- We initialize the values for each state-action pair in the Q-table to 0.

### The Q-Learning algorithm

- Q Learning waits for one interaction, forms a TD target $R_{t+1} + \gamma max_{a} Q(S_{t+1} , a)$, and updates the Q-value $Q(S_{t} , A_{t} )$ for the state-action pair $\left(S_{t} , A_{t} \right) $ in the Q-table using the immediate reward $R_{t+1}$ plus the discounted optimal (i.e., greedy) Q-Value of the following state $\gamma max_{a} Q(S_{t+1} , a)$ scaled by a learning rate $\alpha$.
- The Q-Values in the Q-table become more accurate with more steps.
- **Input:** policy $\pi$, positive integer $num\_episodes$, small positive fraction $\alpha$, $GLIE$ $\{\epsilon_{i}\}$
- **Output:** value function $Q (\approx q_{\pi})$ if num\_episodes is large enough
- **Steps:**
    1. Initialize $Q$ arbitrarily $($e.g. $Q(s,a) = 0$ for all $s \ \epsilon S A(s)$, and $Q(terminal-state, \cdot) = 0 )$
    2. for  $i \leftarrow 1$ to num_episodes
        1. $\epsilon \leftarrow \epsilon_{i}$
        2. Observe $S_{0}$
        3. $t \leftarrow 0$
        4. repeat until $S_{t}$ is terminal
            1. Choose action $A_{t}$ using policy derived from $Q(e.g., \epsilon$-greedy$)$
            2. Take action $A_{t}$  and observe $R_{t+1},S_{t+1}$
            3. $Q(S_{t},A_{t}) \leftarrow Q(S_{t},A_{t}) + \alpha (R_{t+1} + \gamma \cdot max_{a}Q(S_{t+1}, a) - Q(S_{t}, A_{t}))$
            4. $t \leftarrow t + 1$
    3. return $Q$

### Off-policy vs On-policy

- Off-policy refers to using a different policy for acting and updating.
- On-policy refers to using the same policy for acting and updating.


## Lab

* **Objective:** Code a Reinforcement Learning agent from scratch to play [FrozenLake](https://www.gymlibrary.ml/environments/toy_text/frozen_lake/) and [Taxi](https://www.gymlibrary.ml/environments/toy_text/taxi/) using Q-Learning, share it to the community, and experiment with different configurations.
* **Environments:** 
    * [FrozenLake-v1](https://www.gymlibrary.ml/environments/toy_text/frozen_lake/): The agent needs to go from the starting state (S) to the goal state (G) by walking only on frozen tiles (F) and avoiding holes (H).
    * [Taxi-v3](https://www.gymlibrary.ml/environments/toy_text/taxi/): The agent needs to learn to navigate a city to transport its passengers from point A to point B.
* [Syllabus](https://github.com/huggingface/deep-rl-class)
* [Discord server](https://discord.gg/aYka4Yhff9)
* [#study-group-unit2 discord channel](https://discord.gg/aYka4Yhff9)

### Prerequisites
* [Unit 2 README](https://github.com/huggingface/deep-rl-class/blob/main/unit2/README.md)
* [An Introduction to Q-Learning Part 1](https://huggingface.co/blog/deep-rl-q-part1)
* [An Introduction to Q-Learning Part 2](https://huggingface.co/blog/deep-rl-q-part2)

### Objectives
* Be able to use Gym, the environment library.
* Be able to code a Q-Learning agent from scratch.
* Be able to push your trained agent and the code to the Hub with a video replay and an evaluation score.

**Create and run a virual screen**


```python
# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
```




```text
<pyvirtualdisplay.display.Display at 0x7fea103d3b20>
```



**Import the packages**
- `random`: To generate random numbers (that will be useful for Epsilon-Greedy Policy).
- `imageio`: To generate a replay video


```python
import numpy as np
import gym
import random
import imageio
import os

import pickle5 as pickle
```

```text
/home/innom-dt/mambaforge/envs/hf-drl-class-unit2/lib/python3.9/site-packages/gym/envs/registration.py:398: UserWarning: [33mWARN: Custom namespace `ALE` is being overridden by namespace `ALE`. If you are developing a plugin you shouldn't specify a namespace in `register` calls. The namespace is specified through the entry point package metadata.[0m
  logger.warn(
```


### Create and understand [FrozenLake environment â](https://www.gymlibrary.ml/environments/toy_text/frozen_lake/)

* [Documentation](https://www.gymlibrary.ml/environments/toy_text/frozen_lake/)
* The Q-Learning agent needs to navigate from the starting state (S) to the goal state (G) by walking only on frozen tiles (F) and avoid holes (H).
* We can have two sizes of environment:
    * `map_name="4x4"`: a 4x4 grid version
    * `map_name="8x8"`: a 8x8 grid version
* The environment has two modes:
    * `is_slippery=False`: The agent always moves in the intended direction due to the non-slippery nature of the frozen lake.
    * `is_slippery=True`: The agent may not always move in the intended direction due to the slippery nature of the frozen lake (stochastic).


**Create a FrozenLake-v1 environment with a 4x4 non-slippery map**


```python
env = gym.make("FrozenLake-v1", map_name=f"4x4", is_slippery=False)
```

**(Optional)** Define a custom grid:
* "S": start position
* "F": frozen tile
* "H": hole tile
* "G": gift tile

```python
# Custom 4x4 grid
desc=["SFFF", "FHFH", "FFFH", "HFFG"]
gym.make('FrozenLake-v1', desc=desc, is_slippery=True)
```
**Note:** This custom grid arrangement would like like the map below.



### Inspect the environment


```python
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space", env.observation_space)
print("Sample observation", env.observation_space.sample()) # Get a random observation
```

```text
_____OBSERVATION SPACE_____ 

Observation Space Discrete(16)
Sample observation 9
```


**Note:** The observation is a value representing the agentâs current position as $current\_row \cdot nrows + current\_col$, where both the row and col start at 0. 


```python
print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action
```


```text
 _____ACTION SPACE_____ 

Action Space Shape 4
Action Space Sample 3
```


**Action Space:**
* 0: GO LEFT
* 1: GO DOWN
* 2: GO RIGHT
* 3: GO UP

**Reward Function:**
* Reach goal: +1
* Reach hole: 0
* Reach frozen: 0


### Create and Initialize the Q-table


```python
state_space = env.observation_space.n
action_space = env.action_space.n
print(f"There are {state_space} possible states and {action_space} possible actions")
```

```text
There are 16 possible states and 4 possible actions
```

**Define a function to initialize a Q-table**


```python
def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))
```


```python
import pandas as pd
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
```


```python
def display_qtable(qtable, actions, num_rows, num_cols):
    indices = np.array(np.meshgrid(*np.indices((num_rows, num_cols), sparse=True))).T.reshape(-1, 2)
    map_coords = [f"({r},{c})" for r,c in indices]
    return pd.DataFrame(qtable, index=map_coords, columns=actions)
```


```python
Qtable_frozenlake = initialize_q_table(state_space, action_space)
action_names = ['Left', 'Down', 'Right', 'Up']
display_qtable(Qtable_frozenlake, action_names, 4, 4)
```




<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Left</th>
      <th>Down</th>
      <th>Right</th>
      <th>Up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(0,0)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(0,1)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(0,2)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(0,3)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(1,0)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(1,1)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(1,2)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(1,3)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(2,0)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(2,1)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(2,2)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(2,3)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(3,0)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(3,1)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(3,2)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(3,3)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


### Define the greedy policy


```python
def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    return np.argmax(Qtable[state])
```

### Define the epsilon-greedy policy


```python
def epsilon_greedy_policy(Qtable, state):
    # Generate a random number in the interval [0, 1)
    random_num = random.random()
    # if random_num > greater than epsilon --> exploitation, else --> exploration
    return greedy_policy(Qtable, state) if random_num > epsilon else env.action_space.sample()
```

### Define the hyperparameters
* We can use a progressive decay of the epsilon to make sure our agent explores enough of the state space to learn a good value approximation.
* Decreasing the epsilon too quickly might cause the agent to get stuck by not exploring enough of the state space.


```python
# Training parameters
n_training_episodes = 10000  # Total training episodes
learning_rate = 0.7          # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"     # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate
eval_seed = []               # The evaluation seed of the environment

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability 
decay_rate = 0.005            # Exponential decay rate for exploration prob
```

### Create the training loop method


```python
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in range(n_training_episodes):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # Reset the environment
        state = env.reset()
        step = 0
        done = False
        
        # repeat
        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(Qtable, state)
            
            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            td_target = reward + gamma * np.max(Qtable[new_state])
            Qtable[state][action] = Qtable[state][action] + learning_rate * (td_target - Qtable[state][action])
            
            # If done, finish the episode
            if done:
                break
            
            # Our state is the new state
            state = new_state
    return Qtable
```

### Train the Q-Learning agent


```python
Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)
```


### Inspect the updated Q-Learning table


```python
display_qtable(Qtable_frozenlake, action_names, 4, 4)
```




<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Left</th>
      <th>Down</th>
      <th>Right</th>
      <th>Up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(0,0)</th>
      <td>0.735092</td>
      <td>0.773781</td>
      <td>0.773781</td>
      <td>0.735092</td>
    </tr>
    <tr>
      <th>(0,1)</th>
      <td>0.735092</td>
      <td>0.000000</td>
      <td>0.814506</td>
      <td>0.773781</td>
    </tr>
    <tr>
      <th>(0,2)</th>
      <td>0.773781</td>
      <td>0.857375</td>
      <td>0.773781</td>
      <td>0.814506</td>
    </tr>
    <tr>
      <th>(0,3)</th>
      <td>0.814506</td>
      <td>0.000000</td>
      <td>0.773781</td>
      <td>0.773781</td>
    </tr>
    <tr>
      <th>(1,0)</th>
      <td>0.773781</td>
      <td>0.814506</td>
      <td>0.000000</td>
      <td>0.735092</td>
    </tr>
    <tr>
      <th>(1,1)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(1,2)</th>
      <td>0.000000</td>
      <td>0.902500</td>
      <td>0.000000</td>
      <td>0.814506</td>
    </tr>
    <tr>
      <th>(1,3)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(2,0)</th>
      <td>0.814506</td>
      <td>0.000000</td>
      <td>0.857375</td>
      <td>0.773781</td>
    </tr>
    <tr>
      <th>(2,1)</th>
      <td>0.814506</td>
      <td>0.902500</td>
      <td>0.902500</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(2,2)</th>
      <td>0.857375</td>
      <td>0.950000</td>
      <td>0.000000</td>
      <td>0.857375</td>
    </tr>
    <tr>
      <th>(2,3)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(3,0)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(3,1)</th>
      <td>0.000000</td>
      <td>0.902500</td>
      <td>0.950000</td>
      <td>0.857375</td>
    </tr>
    <tr>
      <th>(3,2)</th>
      <td>0.902500</td>
      <td>0.950000</td>
      <td>1.000000</td>
      <td>0.902500</td>
    </tr>
    <tr>
      <th>(3,3)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Define the evaluation method


```python
def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        if seed:
            state = env.reset(seed=seed[episode])
        else:
            state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0
    
        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(Q[state][:])
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward
        
            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward
```

### Evaluate theQ-Learning agent


```python
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
```

```text
Mean_reward=1.00 +/- 0.00
```


**Note:**
* The mean reward should be 1.0
* Try using the [slippery version](https://www.gymlibrary.ml/environments/toy_text/frozen_lake/) of the map.


### Publish our trained model on the Hub


```python
%%capture
from huggingface_hub import HfApi, HfFolder, Repository
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import json
```


```python
def record_video(env, Qtable, out_directory, fps=1):
    images = []
    done = False
    state = env.reset(seed=random.randint(0,500))
    img = env.render(mode='rgb_array')
    images.append(img)
    while not done:
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Qtable[state][:])
        state, reward, done, info = env.step(action) # We directly put next_state = state for recording logic
        img = env.render(mode='rgb_array')
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)
```


**[Leaderboard](https://huggingface.co/spaces/chrisjay/Deep-Reinforcement-Learning-Leaderboard)**

**Log into Hugging Face account**


```python
from huggingface_hub import notebook_login
notebook_login()
```

```text
Login successful
Your token has been saved to /home/innom-dt/.huggingface/token
```


**Create a model dictionnary that contains the hyperparameters and the Q_table**


```python
model = {
    "env_id": env_id,
    "max_steps": max_steps,
    "n_training_episodes": n_training_episodes,
    "n_eval_episodes": n_eval_episodes,
    "eval_seed": eval_seed,

    "learning_rate": learning_rate,
    "gamma": gamma,

    "epsilon": epsilon,
    "max_epsilon": max_epsilon,
    "min_epsilon": min_epsilon,
    "decay_rate": decay_rate,

    "qtable": Qtable_frozenlake
}
model
```


```text
{'env_id': 'FrozenLake-v1',
 'max_steps': 99,
 'n_training_episodes': 10000,
 'n_eval_episodes': 100,
 'eval_seed': [],
 'learning_rate': 0.7,
 'gamma': 0.95,
 'epsilon': 1.0,
 'max_epsilon': 1.0,
 'min_epsilon': 0.05,
 'decay_rate': 0.005,
 'qtable': array([[0.73509189, 0.77378094, 0.77378094, 0.73509189],
        [0.73509189, 0.        , 0.81450625, 0.77378094],
        [0.77378094, 0.857375  , 0.77378094, 0.81450625],
        [0.81450625, 0.        , 0.77378094, 0.77378094],
        [0.77378094, 0.81450625, 0.        , 0.73509189],
        [0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.9025    , 0.        , 0.81450625],
        [0.        , 0.        , 0.        , 0.        ],
        [0.81450625, 0.        , 0.857375  , 0.77378094],
        [0.81450625, 0.9025    , 0.9025    , 0.        ],
        [0.857375  , 0.95      , 0.        , 0.857375  ],
        [0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.9025    , 0.95      , 0.857375  ],
        [0.9025    , 0.95      , 1.        , 0.9025    ],
        [0.        , 0.        , 0.        , 0.        ]])}
```



**Publish the trained model on the Hub**


```python
username = "cj-mills"
repo_name = "q-FrozenLake-v1-4x4-noSlippery"
push_to_hub(
    repo_id=f"{username}/{repo_name}",
    model=model,
    env=env)
```

```text
{'env_id': 'FrozenLake-v1', 'max_steps': 99, 'n_training_episodes': 10000, 'n_eval_episodes': 100, 'eval_seed': [], 'learning_rate': 0.7, 'gamma': 0.95, 'epsilon': 1.0, 'max_epsilon': 1.0, 'min_epsilon': 0.05, 'decay_rate': 0.005, 'qtable': array([[0.73509189, 0.77378094, 0.77378094, 0.73509189],
       [0.73509189, 0.        , 0.81450625, 0.77378094],
       [0.77378094, 0.857375  , 0.77378094, 0.81450625],
       [0.81450625, 0.        , 0.77378094, 0.77378094],
       [0.77378094, 0.81450625, 0.        , 0.73509189],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.9025    , 0.        , 0.81450625],
       [0.        , 0.        , 0.        , 0.        ],
       [0.81450625, 0.        , 0.857375  , 0.77378094],
       [0.81450625, 0.9025    , 0.9025    , 0.        ],
       [0.857375  , 0.95      , 0.        , 0.857375  ],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.9025    , 0.95      , 0.857375  ],
       [0.9025    , 0.95      , 1.        , 0.9025    ],
       [0.        , 0.        , 0.        , 0.        ]]), 'map_name': '4x4', 'slippery': False}
Pushing repo q-FrozenLake-v1-4x4-noSlippery to the Hugging Face Hub
Your model is pushed to the hub. You can view your model here: https://huggingface.co/cj-mills/q-FrozenLake-v1-4x4-noSlippery
```

### Create and understand [Taxi-v3](https://www.gymlibrary.ml/environments/toy_text/taxi/)

* [Documentation](https://www.gymlibrary.ml/environments/toy_text/taxi/)
* There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue).
* The taxi starts off at a random square and the passenger is at a random location. 
* The taxi drives to the passengerâs location, picks up the passenger, drives to the passengerâs destination (another one of the four specified locations), and then drops off the passenger.
* The episode ends once the taxi drops off the passenger.

![taxi.gif](https://www.gymlibrary.ml/_images/taxi.gif)


```python
env = gym.make("Taxi-v3")
```

**Note:** There are 25 taxi positions, five possible passenger locations (including when the passenger is in the taxi), and four destination locations, meaning 500 discrete states.


```python
state_space = env.observation_space.n
action_space = env.action_space.n
print(f"There are {state_space} possible states and {action_space} possible actions")
```

    There are 500 possible states and 6 possible actions


**Action space:**
* 0: move south
* 1: move north
* 2: move east
* 3: move west
* 4: pickup passenger
* 5: drop off passenger

**Reward function:**
* -1 per step unless other reward is triggered.
* +20 delivering passenger.
* -10 executing pickup and drop-off actions illegally.


```python
# Create our Q table with state_size rows and action_size columns (500x6)
Qtable_taxi = initialize_q_table(state_space, action_space)
print("Q-table shape: ", Qtable_taxi .shape)
indices = np.array(np.meshgrid(*np.indices((25, 5, 4), sparse=True))).T.reshape(-1, 3)
map_coords = [f"TaxiPos: {tp}, PassLoc: {pl}, DestLoc: {dl}" for tp,pl,dl in indices]
action_names = ['move south', 'move north', 'move east', 'move west', 'pickup passenger', 'drop off passenger']
pd.DataFrame(Qtable_taxi, index=map_coords, columns=action_names)
```

    Q-table shape:  (500, 6)





<div style="overflow-x:auto; overflow-y:auto; height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>move south</th>
      <th>move north</th>
      <th>move east</th>
      <th>move west</th>
      <th>pickup passenger</th>
      <th>drop off passenger</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TaxiPos: 0, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 0, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 1, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 2, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 3, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 4, DestLoc: 0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 0, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 1, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 2, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 3, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 4, DestLoc: 1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 0, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 1, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 2, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 3, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 4, DestLoc: 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 0, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 1, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 2, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 3, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 4, DestLoc: 3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Define the hyperparameters


```python
# Training parameters
n_training_episodes = 25000   # Total training episodes
learning_rate = 0.7           # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# Environment parameters
env_id = "Taxi-v3"           # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05           # Minimum exploration probability 
decay_rate = 0.005            # Exponential decay rate for exploration prob
```

**DO NOT MODIFY EVAL_SEED**


```python
# DO NOT MODIFY EVAL_SEED
eval_seed = [16,54,165,177,191,191,120,80,149,178,48,38,6,125,174,73,50,172,100,148,146,6,25,40,68,148,49,167,9,97,164,176,61,7,54,55,
 161,131,184,51,170,12,120,113,95,126,51,98,36,135,54,82,45,95,89,59,95,124,9,113,58,85,51,134,121,169,105,21,30,11,50,65,12,43,82,145,152,97,106,55,31,85,38,
 112,102,168,123,97,21,83,158,26,80,63,5,81,32,11,28,148] # Evaluation seed, this ensures that all classmates agents are trained on the same taxi starting position
                                                          # Each seed has a specific starting state
```

### Train a Q-Learning agent


```python
Qtable_taxi = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_taxi)
indices = np.array(np.meshgrid(*np.indices((25, 5, 4), sparse=True))).T.reshape(-1, 3)
map_coords = [f"TaxiPos: {tp}, PassLoc: {pl}, DestLoc: {dl}" for tp,pl,dl in indices]
action_names = ['move south', 'move north', 'move east', 'move west', 'pickup passenger', 'drop off passenger']
pd.DataFrame(Qtable_taxi, index=map_coords, columns=action_names)
```



<div style="overflow-x:auto; overflow-y:auto; height:500px">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>move south</th>
      <th>move north</th>
      <th>move east</th>
      <th>move west</th>
      <th>pickup passenger</th>
      <th>drop off passenger</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TaxiPos: 0, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 1, DestLoc: 0</th>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 2, DestLoc: 0</th>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 3, DestLoc: 0</th>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 4, DestLoc: 0</th>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-13.111427</td>
      <td>-13.111427</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 1, DestLoc: 0</th>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-13.111427</td>
      <td>-13.111427</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 2, DestLoc: 0</th>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 3, DestLoc: 0</th>
      <td>5.209976</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 4, DestLoc: 0</th>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 1, DestLoc: 0</th>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 2, DestLoc: 0</th>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 3, DestLoc: 0</th>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 4, DestLoc: 0</th>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 1, DestLoc: 0</th>
      <td>16.100000</td>
      <td>18.000000</td>
      <td>16.100000</td>
      <td>18.000000</td>
      <td>9.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 2, DestLoc: 0</th>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>-3.790024</td>
      <td>3.949478</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 3, DestLoc: 0</th>
      <td>12.580250</td>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>1.951237</td>
      <td>9.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 4, DestLoc: 0</th>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>-2.463183</td>
      <td>5.209976</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 1, DestLoc: 0</th>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 2, DestLoc: 0</th>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 3, DestLoc: 0</th>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 4, DestLoc: 0</th>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 1, DestLoc: 0</th>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 2, DestLoc: 0</th>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 3, DestLoc: 0</th>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 4, DestLoc: 0</th>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 1, DestLoc: 0</th>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 2, DestLoc: 0</th>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 3, DestLoc: 0</th>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 4, DestLoc: 0</th>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 1, DestLoc: 0</th>
      <td>14.295000</td>
      <td>16.100000</td>
      <td>16.100000</td>
      <td>18.000000</td>
      <td>7.100000</td>
      <td>7.100000</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 2, DestLoc: 0</th>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 3, DestLoc: 0</th>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 4, DestLoc: 0</th>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 1, DestLoc: 0</th>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 2, DestLoc: 0</th>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 3, DestLoc: 0</th>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 4, DestLoc: 0</th>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 1, DestLoc: 0</th>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 2, DestLoc: 0</th>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>-3.790024</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 3, DestLoc: 0</th>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 4, DestLoc: 0</th>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 1, DestLoc: 0</th>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 2, DestLoc: 0</th>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 3, DestLoc: 0</th>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 4, DestLoc: 0</th>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 1, DestLoc: 0</th>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 2, DestLoc: 0</th>
      <td>12.580250</td>
      <td>14.295000</td>
      <td>16.100000</td>
      <td>14.295000</td>
      <td>5.295000</td>
      <td>5.295000</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 3, DestLoc: 0</th>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 4, DestLoc: 0</th>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 1, DestLoc: 0</th>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-2.394933</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 2, DestLoc: 0</th>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 3, DestLoc: 0</th>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-1.468351</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 4, DestLoc: 0</th>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 1, DestLoc: 0</th>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 2, DestLoc: 0</th>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>5.209976</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 3, DestLoc: 0</th>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 4, DestLoc: 0</th>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-2.394933</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 1, DestLoc: 0</th>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-1.468351</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 2, DestLoc: 0</th>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 3, DestLoc: 0</th>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 4, DestLoc: 0</th>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 1, DestLoc: 0</th>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>7.933492</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 2, DestLoc: 0</th>
      <td>14.295000</td>
      <td>16.100000</td>
      <td>18.000000</td>
      <td>14.295000</td>
      <td>7.100000</td>
      <td>7.100000</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 3, DestLoc: 0</th>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>7.933492</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 4, DestLoc: 0</th>
      <td>12.580250</td>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>9.403676</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 1, DestLoc: 0</th>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-13.111427</td>
      <td>-13.111427</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 2, DestLoc: 0</th>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 3, DestLoc: 0</th>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 4, DestLoc: 0</th>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>5.209976</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 1, DestLoc: 0</th>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>5.209976</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 2, DestLoc: 0</th>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>9.403676</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 3, DestLoc: 0</th>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 4, DestLoc: 0</th>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-13.111427</td>
      <td>-13.111427</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 1, DestLoc: 0</th>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 2, DestLoc: 0</th>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 3, DestLoc: 0</th>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 4, DestLoc: 0</th>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 1, DestLoc: 0</th>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-3.790024</td>
      <td>3.949478</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 2, DestLoc: 0</th>
      <td>16.100000</td>
      <td>18.000000</td>
      <td>18.000000</td>
      <td>16.100000</td>
      <td>9.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 3, DestLoc: 0</th>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-3.790024</td>
      <td>3.949478</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 4, DestLoc: 0</th>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>0.403676</td>
      <td>7.933492</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 1, DestLoc: 0</th>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 2, DestLoc: 0</th>
      <td>6.536817</td>
      <td>9.403676</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 3, DestLoc: 0</th>
      <td>2.752004</td>
      <td>5.209976</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 4, DestLoc: 0</th>
      <td>-2.394933</td>
      <td>-4.111427</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 1, DestLoc: 0</th>
      <td>-2.394933</td>
      <td>-4.111427</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 2, DestLoc: 0</th>
      <td>0.533683</td>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 3, DestLoc: 0</th>
      <td>6.536817</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>-3.790024</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 4, DestLoc: 0</th>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 1, DestLoc: 0</th>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 2, DestLoc: 0</th>
      <td>-0.493001</td>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 3, DestLoc: 0</th>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 4, DestLoc: 0</th>
      <td>-0.493001</td>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 1, DestLoc: 0</th>
      <td>14.295000</td>
      <td>18.000000</td>
      <td>14.295000</td>
      <td>16.100000</td>
      <td>7.100000</td>
      <td>7.100000</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 2, DestLoc: 0</th>
      <td>7.933492</td>
      <td>5.209976</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 3, DestLoc: 0</th>
      <td>14.295000</td>
      <td>10.951237</td>
      <td>10.951237</td>
      <td>12.580250</td>
      <td>3.580250</td>
      <td>3.580250</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 4, DestLoc: 0</th>
      <td>9.403676</td>
      <td>6.536817</td>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 0, DestLoc: 0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 1, DestLoc: 0</th>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 2, DestLoc: 0</th>
      <td>5.209976</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 3, DestLoc: 0</th>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 4, DestLoc: 0</th>
      <td>-1.468351</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 1, DestLoc: 1</th>
      <td>-1.468351</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 2, DestLoc: 1</th>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 3, DestLoc: 1</th>
      <td>5.209976</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 4, DestLoc: 1</th>
      <td>0.533683</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 1, DestLoc: 1</th>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 2, DestLoc: 1</th>
      <td>0.533683</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 3, DestLoc: 1</th>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 4, DestLoc: 1</th>
      <td>0.533683</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 1, DestLoc: 1</th>
      <td>12.580250</td>
      <td>16.100000</td>
      <td>14.295000</td>
      <td>16.100000</td>
      <td>5.295000</td>
      <td>5.295000</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 2, DestLoc: 1</th>
      <td>9.403676</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 3, DestLoc: 1</th>
      <td>12.580250</td>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>12.580250</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 4, DestLoc: 1</th>
      <td>10.951237</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 1, DestLoc: 1</th>
      <td>-0.493001</td>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 2, DestLoc: 1</th>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 3, DestLoc: 1</th>
      <td>0.533683</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 4, DestLoc: 1</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 1, DestLoc: 1</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 2, DestLoc: 1</th>
      <td>2.752004</td>
      <td>5.209976</td>
      <td>5.209976</td>
      <td>3.949478</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 3, DestLoc: 1</th>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 4, DestLoc: 1</th>
      <td>-0.493001</td>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 1, DestLoc: 1</th>
      <td>0.533683</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 2, DestLoc: 1</th>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 3, DestLoc: 1</th>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 4, DestLoc: 1</th>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 1, DestLoc: 1</th>
      <td>10.951237</td>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 2, DestLoc: 1</th>
      <td>10.951237</td>
      <td>14.295000</td>
      <td>14.295000</td>
      <td>12.580250</td>
      <td>3.580250</td>
      <td>3.580250</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 3, DestLoc: 1</th>
      <td>10.951237</td>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 4, DestLoc: 1</th>
      <td>12.580250</td>
      <td>9.403676</td>
      <td>12.580250</td>
      <td>10.951237</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 1, DestLoc: 1</th>
      <td>-1.468351</td>
      <td>-3.275187</td>
      <td>-3.275187</td>
      <td>-1.468351</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 2, DestLoc: 1</th>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 3, DestLoc: 1</th>
      <td>-0.493001</td>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 4, DestLoc: 1</th>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 1, DestLoc: 1</th>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 2, DestLoc: 1</th>
      <td>3.949478</td>
      <td>6.536817</td>
      <td>6.536817</td>
      <td>3.949478</td>
      <td>-3.790024</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 3, DestLoc: 1</th>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 4, DestLoc: 1</th>
      <td>-1.468351</td>
      <td>-3.275187</td>
      <td>-3.275187</td>
      <td>-1.468351</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 1, DestLoc: 1</th>
      <td>-0.493001</td>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 2, DestLoc: 1</th>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 3, DestLoc: 1</th>
      <td>5.209976</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 4, DestLoc: 1</th>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 1, DestLoc: 1</th>
      <td>9.403676</td>
      <td>6.536817</td>
      <td>6.536817</td>
      <td>9.403676</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 2, DestLoc: 1</th>
      <td>12.580250</td>
      <td>16.100000</td>
      <td>16.100000</td>
      <td>12.580250</td>
      <td>5.295000</td>
      <td>5.295000</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 3, DestLoc: 1</th>
      <td>9.403676</td>
      <td>6.536817</td>
      <td>6.536817</td>
      <td>9.403676</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 4, DestLoc: 1</th>
      <td>14.295000</td>
      <td>10.951237</td>
      <td>10.951237</td>
      <td>10.951237</td>
      <td>3.580250</td>
      <td>3.580250</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 1, DestLoc: 1</th>
      <td>-2.394933</td>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 2, DestLoc: 1</th>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 3, DestLoc: 1</th>
      <td>-1.468351</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 4, DestLoc: 1</th>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 1, DestLoc: 1</th>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 2, DestLoc: 1</th>
      <td>5.209976</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 3, DestLoc: 1</th>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 4, DestLoc: 1</th>
      <td>-2.394933</td>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 1, DestLoc: 1</th>
      <td>-1.468351</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 2, DestLoc: 1</th>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 3, DestLoc: 1</th>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 4, DestLoc: 1</th>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 1, DestLoc: 1</th>
      <td>7.933492</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 2, DestLoc: 1</th>
      <td>14.295000</td>
      <td>18.000000</td>
      <td>16.100000</td>
      <td>14.295000</td>
      <td>7.100000</td>
      <td>7.100000</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 3, DestLoc: 1</th>
      <td>7.933492</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 4, DestLoc: 1</th>
      <td>12.580250</td>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>12.580250</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 1, DestLoc: 1</th>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 2, DestLoc: 1</th>
      <td>5.209976</td>
      <td>7.933492</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 3, DestLoc: 1</th>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 4, DestLoc: 1</th>
      <td>-3.275187</td>
      <td>-3.275187</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 1, DestLoc: 1</th>
      <td>-3.275187</td>
      <td>-3.275187</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 2, DestLoc: 1</th>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 3, DestLoc: 1</th>
      <td>7.933492</td>
      <td>5.209976</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 4, DestLoc: 1</th>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 1, DestLoc: 1</th>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 2, DestLoc: 1</th>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 3, DestLoc: 1</th>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 4, DestLoc: 1</th>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 1, DestLoc: 1</th>
      <td>12.580250</td>
      <td>16.100000</td>
      <td>12.580250</td>
      <td>14.295000</td>
      <td>5.295000</td>
      <td>5.295000</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 2, DestLoc: 1</th>
      <td>6.536817</td>
      <td>6.536817</td>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 3, DestLoc: 1</th>
      <td>16.100000</td>
      <td>12.580250</td>
      <td>12.580250</td>
      <td>14.295000</td>
      <td>5.295000</td>
      <td>5.295000</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 4, DestLoc: 1</th>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 1, DestLoc: 1</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 2, DestLoc: 1</th>
      <td>3.949478</td>
      <td>6.536817</td>
      <td>3.949478</td>
      <td>6.536817</td>
      <td>-3.790024</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 3, DestLoc: 1</th>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 4, DestLoc: 1</th>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-2.394933</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 1, DestLoc: 1</th>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-2.394933</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 2, DestLoc: 1</th>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 3, DestLoc: 1</th>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>6.536817</td>
      <td>-3.790024</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 4, DestLoc: 1</th>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 1, DestLoc: 1</th>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 2, DestLoc: 1</th>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 3, DestLoc: 1</th>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 4, DestLoc: 1</th>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 1, DestLoc: 1</th>
      <td>10.951237</td>
      <td>14.295000</td>
      <td>10.951237</td>
      <td>14.295000</td>
      <td>3.580250</td>
      <td>3.580250</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 2, DestLoc: 1</th>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>10.951237</td>
      <td>7.933492</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 3, DestLoc: 1</th>
      <td>10.951237</td>
      <td>10.951237</td>
      <td>10.951237</td>
      <td>14.295000</td>
      <td>3.580250</td>
      <td>3.580250</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 4, DestLoc: 1</th>
      <td>9.403676</td>
      <td>9.403676</td>
      <td>12.580250</td>
      <td>9.403676</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 1, DestLoc: 1</th>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 2, DestLoc: 1</th>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>5.209976</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 3, DestLoc: 1</th>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 4, DestLoc: 1</th>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-1.468351</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 0, DestLoc: 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 1, DestLoc: 1</th>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-1.468351</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 2, DestLoc: 1</th>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 3, DestLoc: 1</th>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>5.209976</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 4, DestLoc: 1</th>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 1, DestLoc: 2</th>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 2, DestLoc: 2</th>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 3, DestLoc: 2</th>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>5.209976</td>
      <td>2.752004</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 4, DestLoc: 2</th>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 1, DestLoc: 2</th>
      <td>9.403676</td>
      <td>9.403676</td>
      <td>9.403676</td>
      <td>12.580250</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 2, DestLoc: 2</th>
      <td>9.403676</td>
      <td>12.580250</td>
      <td>12.580250</td>
      <td>9.403676</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 3, DestLoc: 2</th>
      <td>9.403676</td>
      <td>9.403676</td>
      <td>9.403676</td>
      <td>12.580250</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 4, DestLoc: 2</th>
      <td>10.951237</td>
      <td>10.951237</td>
      <td>14.295000</td>
      <td>10.951237</td>
      <td>3.580250</td>
      <td>3.580250</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 1, DestLoc: 2</th>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 2, DestLoc: 2</th>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 3, DestLoc: 2</th>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 4, DestLoc: 2</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 1, DestLoc: 2</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 2, DestLoc: 2</th>
      <td>2.752004</td>
      <td>5.209976</td>
      <td>5.209976</td>
      <td>2.752004</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 3, DestLoc: 2</th>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 4, DestLoc: 2</th>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 1, DestLoc: 2</th>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 2, DestLoc: 2</th>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 3, DestLoc: 2</th>
      <td>6.536817</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>-3.790024</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 4, DestLoc: 2</th>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 1, DestLoc: 2</th>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>10.951237</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 2, DestLoc: 2</th>
      <td>10.951237</td>
      <td>14.295000</td>
      <td>14.295000</td>
      <td>10.951237</td>
      <td>3.580250</td>
      <td>3.580250</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 3, DestLoc: 2</th>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>10.951237</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 4, DestLoc: 2</th>
      <td>16.100000</td>
      <td>12.580250</td>
      <td>12.580250</td>
      <td>12.580250</td>
      <td>5.295000</td>
      <td>5.295000</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 1, DestLoc: 2</th>
      <td>-3.275187</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 2, DestLoc: 2</th>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 3, DestLoc: 2</th>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 4, DestLoc: 2</th>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 1, DestLoc: 2</th>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 2, DestLoc: 2</th>
      <td>3.949478</td>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>3.949478</td>
      <td>-3.790024</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 3, DestLoc: 2</th>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 4, DestLoc: 2</th>
      <td>-3.275187</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 1, DestLoc: 2</th>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 2, DestLoc: 2</th>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 3, DestLoc: 2</th>
      <td>5.209976</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 4, DestLoc: 2</th>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 1, DestLoc: 2</th>
      <td>6.536817</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 2, DestLoc: 2</th>
      <td>12.580250</td>
      <td>16.100000</td>
      <td>14.295000</td>
      <td>12.580250</td>
      <td>5.295000</td>
      <td>5.295000</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 3, DestLoc: 2</th>
      <td>6.536817</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 4, DestLoc: 2</th>
      <td>14.295000</td>
      <td>10.951237</td>
      <td>12.580250</td>
      <td>14.295000</td>
      <td>3.580250</td>
      <td>3.580250</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 1, DestLoc: 2</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 2, DestLoc: 2</th>
      <td>3.949478</td>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>5.209976</td>
      <td>-3.790024</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 3, DestLoc: 2</th>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 4, DestLoc: 2</th>
      <td>-4.111427</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-3.275187</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 1, DestLoc: 2</th>
      <td>-4.111427</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-3.275187</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 2, DestLoc: 2</th>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 3, DestLoc: 2</th>
      <td>9.403676</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 4, DestLoc: 2</th>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 1, DestLoc: 2</th>
      <td>5.209976</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 2, DestLoc: 2</th>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 3, DestLoc: 2</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 4, DestLoc: 2</th>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 1, DestLoc: 2</th>
      <td>10.951237</td>
      <td>14.295000</td>
      <td>12.580250</td>
      <td>12.580250</td>
      <td>3.580250</td>
      <td>3.580250</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 2, DestLoc: 2</th>
      <td>5.209976</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>6.536817</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 3, DestLoc: 2</th>
      <td>18.000000</td>
      <td>14.295000</td>
      <td>16.100000</td>
      <td>16.100000</td>
      <td>7.100000</td>
      <td>7.100000</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 4, DestLoc: 2</th>
      <td>6.536817</td>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 1, DestLoc: 2</th>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 2, DestLoc: 2</th>
      <td>2.752004</td>
      <td>5.209976</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 3, DestLoc: 2</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 4, DestLoc: 2</th>
      <td>-3.275187</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 1, DestLoc: 2</th>
      <td>-3.275187</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 2, DestLoc: 2</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 3, DestLoc: 2</th>
      <td>2.752004</td>
      <td>5.209976</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 4, DestLoc: 2</th>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 1, DestLoc: 2</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 2, DestLoc: 2</th>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 3, DestLoc: 2</th>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 4, DestLoc: 2</th>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 1, DestLoc: 2</th>
      <td>9.403676</td>
      <td>12.580250</td>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 2, DestLoc: 2</th>
      <td>6.536817</td>
      <td>9.403676</td>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 3, DestLoc: 2</th>
      <td>9.403676</td>
      <td>12.580250</td>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 4, DestLoc: 2</th>
      <td>7.933492</td>
      <td>10.951237</td>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 1, DestLoc: 2</th>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 2, DestLoc: 2</th>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 3, DestLoc: 2</th>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 4, DestLoc: 2</th>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 1, DestLoc: 2</th>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 2, DestLoc: 2</th>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 3, DestLoc: 2</th>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 4, DestLoc: 2</th>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 1, DestLoc: 2</th>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 2, DestLoc: 2</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 3, DestLoc: 2</th>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 4, DestLoc: 2</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 1, DestLoc: 2</th>
      <td>7.933492</td>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 2, DestLoc: 2</th>
      <td>7.933492</td>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 3, DestLoc: 2</th>
      <td>7.933492</td>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 4, DestLoc: 2</th>
      <td>9.403676</td>
      <td>12.580250</td>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 1, DestLoc: 2</th>
      <td>-3.275187</td>
      <td>-1.468351</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 2, DestLoc: 2</th>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 3, DestLoc: 2</th>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 4, DestLoc: 2</th>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 1, DestLoc: 2</th>
      <td>-1.468351</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 2, DestLoc: 2</th>
      <td>1.614404</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 3, DestLoc: 2</th>
      <td>0.533683</td>
      <td>2.752004</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 4, DestLoc: 2</th>
      <td>-3.275187</td>
      <td>-1.468351</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 0, DestLoc: 2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 1, DestLoc: 2</th>
      <td>-2.394933</td>
      <td>-0.493001</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 2, DestLoc: 2</th>
      <td>5.209976</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 3, DestLoc: 2</th>
      <td>7.933492</td>
      <td>5.209976</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 4, DestLoc: 2</th>
      <td>5.209976</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 1, DestLoc: 3</th>
      <td>6.536817</td>
      <td>9.403676</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 2, DestLoc: 3</th>
      <td>9.403676</td>
      <td>12.580250</td>
      <td>12.580250</td>
      <td>10.951237</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 3, DestLoc: 3</th>
      <td>6.536817</td>
      <td>9.403676</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 0, PassLoc: 4, DestLoc: 3</th>
      <td>18.000000</td>
      <td>14.295000</td>
      <td>14.295000</td>
      <td>16.100000</td>
      <td>7.100000</td>
      <td>7.100000</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 1, DestLoc: 3</th>
      <td>-4.111427</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 2, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 3, DestLoc: 3</th>
      <td>-3.275187</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 1, PassLoc: 4, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 1, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 2, DestLoc: 3</th>
      <td>2.752004</td>
      <td>5.209976</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 3, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 2, PassLoc: 4, DestLoc: 3</th>
      <td>-4.111427</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 1, DestLoc: 3</th>
      <td>-3.275187</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 2, DestLoc: 3</th>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 3, DestLoc: 3</th>
      <td>6.536817</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-3.790024</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 3, PassLoc: 4, DestLoc: 3</th>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 1, DestLoc: 3</th>
      <td>5.209976</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 2, DestLoc: 3</th>
      <td>10.951237</td>
      <td>14.295000</td>
      <td>12.580250</td>
      <td>10.951237</td>
      <td>3.580250</td>
      <td>3.580250</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 3, DestLoc: 3</th>
      <td>5.209976</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 4, PassLoc: 4, DestLoc: 3</th>
      <td>16.100000</td>
      <td>12.580250</td>
      <td>14.295000</td>
      <td>16.100000</td>
      <td>5.295000</td>
      <td>5.295000</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 1, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 2, DestLoc: 3</th>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 3, DestLoc: 3</th>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 5, PassLoc: 4, DestLoc: 3</th>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-4.111427</td>
      <td>-13.111427</td>
      <td>-13.111427</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 1, DestLoc: 3</th>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-4.111427</td>
      <td>-13.111427</td>
      <td>-13.111427</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 2, DestLoc: 3</th>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 3, DestLoc: 3</th>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 6, PassLoc: 4, DestLoc: 3</th>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 1, DestLoc: 3</th>
      <td>5.209976</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 2, DestLoc: 3</th>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 3, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 7, PassLoc: 4, DestLoc: 3</th>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 1, DestLoc: 3</th>
      <td>10.951237</td>
      <td>12.580250</td>
      <td>10.951237</td>
      <td>10.951237</td>
      <td>1.951237</td>
      <td>9.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 2, DestLoc: 3</th>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>5.209976</td>
      <td>-3.790024</td>
      <td>3.949478</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 3, DestLoc: 3</th>
      <td>18.000000</td>
      <td>16.100000</td>
      <td>18.000000</td>
      <td>18.000000</td>
      <td>9.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 8, PassLoc: 4, DestLoc: 3</th>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>6.536817</td>
      <td>-2.463183</td>
      <td>5.209976</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 1, DestLoc: 3</th>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 2, DestLoc: 3</th>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 3, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 9, PassLoc: 4, DestLoc: 3</th>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 1, DestLoc: 3</th>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 2, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 3, DestLoc: 3</th>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 10, PassLoc: 4, DestLoc: 3</th>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 1, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 2, DestLoc: 3</th>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 3, DestLoc: 3</th>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 11, PassLoc: 4, DestLoc: 3</th>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 1, DestLoc: 3</th>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 2, DestLoc: 3</th>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 3, DestLoc: 3</th>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 12, PassLoc: 4, DestLoc: 3</th>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 1, DestLoc: 3</th>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 2, DestLoc: 3</th>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 3, DestLoc: 3</th>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 13, PassLoc: 4, DestLoc: 3</th>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 1, DestLoc: 3</th>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 2, DestLoc: 3</th>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 3, DestLoc: 3</th>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 14, PassLoc: 4, DestLoc: 3</th>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 1, DestLoc: 3</th>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 2, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 3, DestLoc: 3</th>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>0.533683</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 15, PassLoc: 4, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 1, DestLoc: 3</th>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 2, DestLoc: 3</th>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 3, DestLoc: 3</th>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>-1.066508</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 16, PassLoc: 4, DestLoc: 3</th>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>7.933492</td>
      <td>0.403676</td>
      <td>0.403676</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 1, DestLoc: 3</th>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 2, DestLoc: 3</th>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 3, DestLoc: 3</th>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 17, PassLoc: 4, DestLoc: 3</th>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 1, DestLoc: 3</th>
      <td>-1.468351</td>
      <td>-0.493001</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-10.468351</td>
      <td>-10.468351</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 2, DestLoc: 3</th>
      <td>1.614404</td>
      <td>2.752004</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>-7.385596</td>
      <td>-7.385596</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 3, DestLoc: 3</th>
      <td>0.533683</td>
      <td>1.614404</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-8.466317</td>
      <td>-8.466317</td>
    </tr>
    <tr>
      <th>TaxiPos: 18, PassLoc: 4, DestLoc: 3</th>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 1, DestLoc: 3</th>
      <td>-2.394933</td>
      <td>-1.468351</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-11.394933</td>
      <td>-11.394933</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 2, DestLoc: 3</th>
      <td>5.209976</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 3, DestLoc: 3</th>
      <td>7.933492</td>
      <td>6.536817</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>9.403676</td>
      <td>-1.066508</td>
    </tr>
    <tr>
      <th>TaxiPos: 19, PassLoc: 4, DestLoc: 3</th>
      <td>5.209976</td>
      <td>3.949478</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 1, DestLoc: 3</th>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-2.463183</td>
      <td>5.209976</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 2, DestLoc: 3</th>
      <td>9.403676</td>
      <td>10.951237</td>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>0.403676</td>
      <td>7.933492</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 3, DestLoc: 3</th>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-2.463183</td>
      <td>5.209976</td>
    </tr>
    <tr>
      <th>TaxiPos: 20, PassLoc: 4, DestLoc: 3</th>
      <td>18.000000</td>
      <td>16.100000</td>
      <td>16.100000</td>
      <td>18.000000</td>
      <td>9.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 1, DestLoc: 3</th>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-13.111427</td>
      <td>-13.111427</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 2, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 3, DestLoc: 3</th>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 21, PassLoc: 4, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 1, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>-1.468351</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 2, DestLoc: 3</th>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>1.614404</td>
      <td>-6.247996</td>
      <td>-6.247996</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 3, DestLoc: 3</th>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-0.493001</td>
      <td>0.533683</td>
      <td>-9.493001</td>
      <td>-9.493001</td>
    </tr>
    <tr>
      <th>TaxiPos: 22, PassLoc: 4, DestLoc: 3</th>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-4.111427</td>
      <td>-3.275187</td>
      <td>-13.111427</td>
      <td>-13.111427</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 1, DestLoc: 3</th>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-3.275187</td>
      <td>-2.394933</td>
      <td>-12.275187</td>
      <td>-12.275187</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 2, DestLoc: 3</th>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 3, DestLoc: 3</th>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>7.933492</td>
      <td>-2.463183</td>
      <td>-2.463183</td>
    </tr>
    <tr>
      <th>TaxiPos: 23, PassLoc: 4, DestLoc: 3</th>
      <td>3.949478</td>
      <td>2.752004</td>
      <td>3.949478</td>
      <td>5.209976</td>
      <td>-5.050522</td>
      <td>-5.050522</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 0, DestLoc: 3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 1, DestLoc: 3</th>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-3.790024</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 2, DestLoc: 3</th>
      <td>10.951237</td>
      <td>12.580250</td>
      <td>10.951237</td>
      <td>9.403676</td>
      <td>1.951237</td>
      <td>1.951237</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 3, DestLoc: 3</th>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>5.209976</td>
      <td>6.536817</td>
      <td>-3.790024</td>
      <td>-3.790024</td>
    </tr>
    <tr>
      <th>TaxiPos: 24, PassLoc: 4, DestLoc: 3</th>
      <td>16.100000</td>
      <td>14.295000</td>
      <td>16.100000</td>
      <td>18.000000</td>
      <td>7.100000</td>
      <td>7.100000</td>
    </tr>
  </tbody>
</table>
</div>


### Create a model dictionary


```python
model = {
    "env_id": env_id,
    "max_steps": max_steps,
    "n_training_episodes": n_training_episodes,
    "n_eval_episodes": n_eval_episodes,
    "eval_seed": eval_seed,

    "learning_rate": learning_rate,
    "gamma": gamma,

    "epsilon": epsilon,
    "max_epsilon": max_epsilon,
    "min_epsilon": min_epsilon,
    "decay_rate": decay_rate,

    "qtable": Qtable_taxi
}
```

**Publish the trained model on the Hub**


```python
username = "cj-mills"
repo_name = "q-Taxi-v3"
push_to_hub(
    repo_id=f"{username}/{repo_name}",
    model=model,
    env=env)
```

```text
IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16, resizing from (550, 350) to (560, 352) to ensure video compatibility with most codecs and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1 (risking incompatibility).
```


```text
{'env_id': 'Taxi-v3', 'max_steps': 99, 'n_training_episodes': 25000, 'n_eval_episodes': 100, 'eval_seed': [16, 54, 165, 177, 191, 191, 120, 80, 149, 178, 48, 38, 6, 125, 174, 73, 50, 172, 100, 148, 146, 6, 25, 40, 68, 148, 49, 167, 9, 97, 164, 176, 61, 7, 54, 55, 161, 131, 184, 51, 170, 12, 120, 113, 95, 126, 51, 98, 36, 135, 54, 82, 45, 95, 89, 59, 95, 124, 9, 113, 58, 85, 51, 134, 121, 169, 105, 21, 30, 11, 50, 65, 12, 43, 82, 145, 152, 97, 106, 55, 31, 85, 38, 112, 102, 168, 123, 97, 21, 83, 158, 26, 80, 63, 5, 81, 32, 11, 28, 148], 'learning_rate': 0.7, 'gamma': 0.95, 'epsilon': 1.0, 'max_epsilon': 1.0, 'min_epsilon': 0.05, 'decay_rate': 0.005, 'qtable': array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ],
       [ 2.75200369,  3.94947757,  2.75200369,  3.94947757,  5.20997639,
        -5.05052243],
       [ 7.93349184,  9.40367562,  7.93349184,  9.40367562, 10.9512375 ,
         0.40367562],
       ...,
       [10.9512375 , 12.58025   , 10.9512375 ,  9.40367562,  1.9512375 ,
         1.9512375 ],
       [ 5.20997639,  6.53681725,  5.20997639,  6.53681725, -3.79002361,
        -3.79002361],
       [16.1       , 14.295     , 16.1       , 18.        ,  7.1       ,
         7.1       ]])}
Pushing repo q-Taxi-v3 to the Hugging Face Hub
```


```text
[swscaler @ 0x5936a80] Warning: data is not aligned! This can lead to a speed loss
```

```text
Upload file replay.mp4:  27%|##7       | 32.0k/118k [00:00<?, ?B/s]
```


```text
Your model is pushed to the hub. You can view your model here: https://huggingface.co/cj-mills/q-Taxi-v3
```


**[Leaderboard](https://huggingface.co/spaces/chrisjay/Deep-Reinforcement-Learning-Leaderboard)**

### Load from Hub
1. Go to [https://huggingface.co/models?other=q-learning](https://huggingface.co/models?other=q-learning) to see the list of all the q-learning saved models.
2. Select one and copy its repo_id.
3. Use `load_from_hub` with the repo_id and the filename.

#### Do not modify this code


```python
from urllib.error import HTTPError

from huggingface_hub import hf_hub_download


def load_from_hub(repo_id: str, filename: str) -> str:
    """
    Download a model from Hugging Face Hub.
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param filename: name of the model zip file from the repository
    """
    try:
        from huggingface_hub import cached_download, hf_hub_url
    except ImportError:
        raise ImportError(
            "You need to install huggingface_hub to use `load_from_hub`. "
            "See https://pypi.org/project/huggingface-hub/ for installation."
        )

    # Get the model from the Hub, download and cache the model on your local disk
    pickle_model = hf_hub_download(
        repo_id=repo_id,
        filename=filename
    )

    with open(pickle_model, 'rb') as f:
        downloaded_model_file = pickle.load(f)
    
    return downloaded_model_file
```


```python
model = load_from_hub(repo_id="cj-mills/q-Taxi-v3", filename="q-learning.pkl")

print(model)
env = gym.make(model["env_id"])

evaluate_agent(env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"])
```

```text
{'env_id': 'Taxi-v3', 'max_steps': 99, 'n_training_episodes': 25000, 'n_eval_episodes': 100, 'eval_seed': [16, 54, 165, 177, 191, 191, 120, 80, 149, 178, 48, 38, 6, 125, 174, 73, 50, 172, 100, 148, 146, 6, 25, 40, 68, 148, 49, 167, 9, 97, 164, 176, 61, 7, 54, 55, 161, 131, 184, 51, 170, 12, 120, 113, 95, 126, 51, 98, 36, 135, 54, 82, 45, 95, 89, 59, 95, 124, 9, 113, 58, 85, 51, 134, 121, 169, 105, 21, 30, 11, 50, 65, 12, 43, 82, 145, 152, 97, 106, 55, 31, 85, 38, 112, 102, 168, 123, 97, 21, 83, 158, 26, 80, 63, 5, 81, 32, 11, 28, 148], 'learning_rate': 0.7, 'gamma': 0.95, 'epsilon': 1.0, 'max_epsilon': 1.0, 'min_epsilon': 0.05, 'decay_rate': 0.005, 'qtable': array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ],
       [ 2.75200369,  3.94947757,  2.75200369,  3.94947757,  5.20997639,
        -5.05052243],
       [ 7.93349184,  9.40367562,  7.93349184,  9.40367562, 10.9512375 ,
         0.40367562],
       ...,
       [10.9512375 , 12.58025   , 10.9512375 ,  9.40367562,  1.9512375 ,
         1.9512375 ],
       [ 5.20997639,  6.53681725,  5.20997639,  6.53681725, -3.79002361,
        -3.79002361],
       [16.1       , 14.295     , 16.1       , 18.        ,  7.1       ,
         7.1       ]])}
```

```text
(7.56, 2.706732347314747)
```



### Some additional challenges
* Train for more steps.
* Try different hyperparameters by looking at what your classmates have done.
* Try using FrozenLake-v1 slippery version and other environments.


## References

* [The Hugging Face Deep Reinforcement Learning Class](https://github.com/huggingface/deep-rl-class)
* [An Introduction to Deep Reinforcement Learning](https://huggingface.co/blog/deep-rl-intro)



<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->

