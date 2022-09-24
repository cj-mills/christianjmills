---
categories:
- ai
- huggingface
- reinforcement-learning
- notes
date: 2022-5-5
description: Unit 1 introduces the basic concepts for reinforcement learning and covers
  how to train an agent for the classic lunar lander environment.
hide: false
layout: post
search_exclude: false
comments:
  utterances:
    repo: cj-mills/christianjmills
title: Notes on The Hugging Face Deep RL Class Pt.1
toc: false

aliases:
- /Notes-on-HF-Deep-Reinforcement-Learning-Class-1/
---

* [What is Reinforcement Learning?](#what-is-reinforcement-learning)
* [The Reinforcement Learning Framework](#the-reinforcement-learning-framework)
* [Exploration-Exploitation Tradeoff](#exploration-exploitation-tradeoff)
* [The Policy](#the-policy)
* [Deep Reinforcement Learning](#deep-reinforcement-learning)
* [Lab](#lab)
* [References](#references)



## What is Reinforcement Learning?

* Reinforcement learning (RL) is a framework for solving control tasks where agents learn from the environment by interacting with it through trial and error and receiving rewards as unique feedback.



## The Reinforcement Learning Framework

### The RL Process

* The RL process is a loop that outputs a sequence of state $S_{0}$, action $A_{0}$, reward $R_{1}$, and next state $S_{1}$.

### The Reward Hypothesis

* The reward and next state result from taking the current action in the current state.
* The goal is to maximize the expected cumulative reward, called the expected return.

### Markov Property

* The Markov property implies that agents only need the current state to decide what action to take and not the history of all the states and actions.

### Observation/States Space

* Observations/States are the information agents get from the environment.
* The state is a complete description of the agent's environment (e.g., a chessboard).
* An observation is a partial description of the state (e.g., the current frame of a video game).

### Action Space

* The action space is the set of all possible actions in an environment.
* Actions can be discrete (e.g., up, down, left, right) or continuous (e.g., steering angle).
* Different RL algorithms are suited for discrete and continuous actions.

### Rewards and discounting

* The reward is the only feedback the agent receives for its actions.
* Rewards that happen earlier in a session (e.g., at the beginning of the game) are more probable since they are more predictable than the long-term reward.
* We can discount longer-term reward values that are less predictable.
* We define a discount rate called gamma with a value between 0 and 1. The discount rate is typically 0.99 or 0.95.
* The larger the gamma, the smaller the discount, meaning agents care more about long-term rewards.
* We discount each reward by gamma to the exponent of the time step, so they are less predictable the further into the future.
* We can write the cumulative reward at each time step $t$ as:

### $$R(\tau) = r_{t+1} + r_{t+2} + r_{t+3} + r_{t+4} + \ldots$$

### $$R(\tau) = \sum^{\infty}_{k=0}{r_{t} + k + 1}$$

* Discounted cumulative expected reward:

### $$R(\tau) = r_{t+1} + \gamma r_{t+2} + \gamma^{2}r_{t+3} + \gamma^{3}r_{t+4} + \ldots$$

### $$R(\tau) = \sum^{\infty}_{k=0}{\gamma^k{} r_{t} + k + 1}$$

### Type of tasks

* A task is an instance of a Reinforcement Learning problem and is either episodic or continuous.

#### Episodic Tasks

* Episodic tasks have starting points and ending points.
* We can represent episodes as a list of states, actions, rewards, and new states.

#### Continuous Tasks

* Continuous tasks have no terminal state, and the agent must learn to choose the best actions and simultaneously interact with the environment.



## Exploration-Exploitation Tradeoff

* We must balance gaining more information about the environment and exploiting known information to maximize reward (e.g., going with the usual restaurant or trying a new one).



## The Policy

* The policy is the function that tells the agent what action to take given the current state.
* The goal is to find the optimal policy $\pi$ which maximizes the expected return.

* $a = \pi(s)$

* $\pi\left( a \vert s \right) = P \left[ A \vert s \right]$

* $\text{policy} \left( \text{actions} \ \vert \ \text{state} \right) = \text{probability distribution over the set of actions given the current state}$

### Policy-based Methods

* Policy-based methods involve learning a policy function directly by teaching the agent which action to take in a given state.
* A deterministic policy will always return the same action in a given state.
* A stochastic policy outputs a probability distribution over actions.

### Value-based methods

* Value-based methods teach the agent to learn which future state is more valuable.
* Value-based methods involve training a value function that maps a state to the expected value of being in that state.
* The value of a state is the expected discounted return the agent can get if it starts in that state and then acts according to the policy.



## Deep Reinforcement Learning

* Deep reinforcement learning introduces deep neural networks to solve RL problems.



## Lab

* **Objective:** Train a lander agent to land correctly, share it to the community, and experiment with different configurations.
* [Syllabus](https://github.com/huggingface/deep-rl-class)
* [Discord server](https://discord.gg/aYka4Yhff9)
* [#study-group-unit1 discord channel](https://discord.gg/aYka4Yhff9)
* Environment: [LunarLander-v2](https://www.gymlibrary.ml/environments/box2d/lunar_lander/)
* RL-Library: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)

### Prerequisites

* [Unit 1 README](https://github.com/huggingface/deep-rl-class/blob/main/unit1/README.md)
* [An Introduction to Deep Reinforcement Learning](https://huggingface.co/blog/deep-rl-intro)


### Objectives
- Be able to use **Gym**, the environment library.
- Be able to use **Stable-Baselines3**, the deep reinforcement learning library.
- Be able to **push your trained agent to the Hub** with a nice video replay and an evaluation score.


### Set the GPU (Google Colab)
- `Runtime > Change Runtime type`
- `Hardware Accelerator > GPU`

### Install dependencies

**Install virtual screen libraries for rendering the environment**
```python
%%capture
!apt install python-opengl
!apt install ffmpeg
!apt install xvfb
!pip3 install pyvirtualdisplay
```

------

**Create and run a virual screen**
```python
# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
```
```text
    <pyvirtualdisplay.display.Display at 0x7f2df34855d0>
```

------


#### Gym[box2d]
* Gym is a toolkit that contains test environments for developing and comparing reinforcement learning algorithms.
* [Box2D](https://www.gymlibrary.ml/environments/box2d/) environments all involve toy games based around physics control, using [box2d](https://box2d.org/)-based physics and PyGame-based rendering.
* [GitHub Repository](https://github.com/openai/gym)
* [Gym Documentation](https://www.gymlibrary.ml/)


#### Stable Baselines
* The Stable Baselines3 library is a set of reliable implementations of reinforcement learning algorithms in PyTorch.
* [GitHub Repository](https://github.com/DLR-RM/stable-baselines3)
* [Documentation](https://stable-baselines3.readthedocs.io/en/master/index.html)

#### Hugging Face x Stable-baselines
* Load and upload Stable-baseline3 models from the Hugging Face Hub.
* [GitHub Repository](https://github.com/huggingface/huggingface_sb3)

------

```python
%%capture
!pip install gym[box2d]
!pip install stable-baselines3[extra]
!pip install huggingface_sb3
!pip install ale-py==0.7.4 # To overcome an issue with gym (https://github.com/DLR-RM/stable-baselines3/issues/875)
```

------


### Import the packages
The Hugging Face Hub Hugging Face works as a central place where anyone can share and explore models and datasets. It has versioning, metrics, visualizations and other features that will allow you to easilly collaborate with others.

[Hugging Face Hub Deep reinforcement Learning models](https://huggingface.co/models?pipeline_tag=reinforcement-learning&sort=downloads)

#### `load_from_hub`
* Download a model from Hugging Face Hub.
* [Source Code](https://github.com/huggingface/huggingface_sb3/blob/23837ad2617c4288e1df71551ac2ef7f3eeee9d5/huggingface_sb3/load_from_hub.py#L6)

#### `package_to_hub`
* Evaluate a model, generate a demo video, and upload the model to Hugging Face Hub.
* [Source Code](https://github.com/huggingface/huggingface_sb3/blob/23837ad2617c4288e1df71551ac2ef7f3eeee9d5/huggingface_sb3/push_to_hub.py#L241)

#### `push_to_hub`
* Upload a model to Hugging Face Hub.
* [Source Code](https://github.com/huggingface/huggingface_sb3/blob/23837ad2617c4288e1df71551ac2ef7f3eeee9d5/huggingface_sb3/push_to_hub.py#L350)

#### `notebook_login`
* Display a widget to login to the HF website and store the token.
* [Source Code](https://github.com/huggingface/huggingface_hub/blob/7042df38a9839ac41efd60cf7f985d473c49d426/src/huggingface_hub/commands/user.py#L312)

#### `PPO`
* The [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) algorithm
* [Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

#### `evaluate_policy`
* Run a policy and return the average reward.
* [Documentation](https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html#stable_baselines3.common.evaluation.evaluate_policy)

#### `make_vec_env`
* Create a wrapped, monitored vectorized environment ([VecEnv](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html?highlight=VecEnv#vecenv)).
* [Documentation](https://stable-baselines3.readthedocs.io/en/master/common/env_util.html#stable_baselines3.common.env_util.make_vec_env)


------

```python
import gym

from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
from huggingface_hub import notebook_login

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
```

------

### Understand the Gym API

1. Create our environment using `gym.make()`
2. Reset the environment to its initial state with `observation = env.reset()`
3. Get an action using our model
4. Perform the action using `env.step(action)`, which returns:
- `obsevation`: The new state (st+1)
- `reward`: The reward we get after executing the action
- `done`: Indicates if the episode terminated
- `info`: A dictionary that provides additional environment-specific information.
5. Reset the environment to its initial state with `observation = env.reset()` at the end of each episode


### Create the LunarLander environment and understand how it works

#### Lunar Lander Environment
* This environment is a classic rocket trajectory optimization problem.
* The agent needs to learn **to adapt its speed and position(horizontal, vertical, and angular) to land correctly.**
* [Documentation](https://www.gymlibrary.ml/environments/box2d/lunar_lander/)

|                   |                                             |
| ----------------- | ------------------------------------------- |
| Action Space      | Discrete(4)                                 |
| Observation Space | (8,)                                        |
| Observation High  | `[inf inf inf inf inf inf inf inf]`         |
| Observation Low   | `[-inf -inf -inf -inf -inf -inf -inf -inf]` |
| Import            | `gym.make("LunarLander-v2")`                |

**Create a [Lunar Lander](https://www.gymlibrary.ml/environments/box2d/lunar_lander/) environment**
```python
env = gym.make("LunarLander-v2")
```

------

**Reset the environment**
```python
observation = env.reset()
```

------

**Take some random actions in the environment**
```python
for _ in range(20):
  # Take a random action
  action = env.action_space.sample()
  print("Action taken:", action)

  # Do this action in the environment and get
  # next_state, reward, done and info
  observation, reward, done, info = env.step(action)
  
  # If the game is done (in our case we land, crashed or timeout)
  if done:
      # Reset the environment
      print("Environment is reset")
      observation = env.reset()
```
```text
    Action taken: 0
    Action taken: 1
    Action taken: 0
    Action taken: 3
    Action taken: 0
    Action taken: 3
    Action taken: 1
    Action taken: 1
    Action taken: 0
    Action taken: 1
    Action taken: 0
    Action taken: 1
    Action taken: 0
    Action taken: 2
    Action taken: 1
    Action taken: 2
    Action taken: 3
    Action taken: 3
    Action taken: 3
    Action taken: 3
```

------


**Inspect the observation space**
```python
# We create a new environment
env = gym.make("LunarLander-v2")
# Reset the environment
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # Get a random observation
```
```text
    _____OBSERVATION SPACE_____ 
    
    Observation Space Shape (8,)
    Sample observation [ 1.9953048  -0.9302978   0.26271465 -1.406391    0.42527643 -0.07207114
      2.1984298   0.4171027 ]
```


**Note:**
* The observation is a vector of size 8, where each value is a different piece of information about the lander.
    1. Horizontal pad coordinate (x)
    2. Vertical pad coordinate (y)
    3. Horizontal speed (x)
    4. Vertical speed (y)
    5. Angle
    6. Angular speed
    7. If the left leg has contact point touched the land
    8. If the right leg has contact point touched the land

------


**Inspect the action space**
```python
print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action
```
```text

     _____ACTION SPACE_____ 
    
    Action Space Shape 4
    Action Space Sample 1
```

**Note:**
* The action space is discrete, with four available actions.
    1. Do nothing.
    2. Fire left orientation engine.
    3. Fire the main engine.
    4. Fire right orientation engine.

* Reward function details:
    - Moving from the top of the screen to the landing pad and zero speed is about 100~140 points.
    - Firing main engine is -0.3 each frame
    - Each leg ground contact is +10 points
    - Episode finishes if the lander crashes (additional - 100 points) or come to rest (+100 points)
    - The game is solved if your agent does 200 points.

------

##### Vectorized Environment
- We can stack multiple independent environments into a single vector to get more diverse experiences during the training.

**Stack 16 independent environments**
```python
env = make_vec_env('LunarLander-v2', n_envs=16)
```

### Create the Model

* [PPO (aka Proximal Policy Optimization)](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) is a combination of:
    - *Value-based reinforcement learning method*: learning an action-value function that will tell us what's the **most valuable action to take given a state and action**.
    - *Policy-based reinforcement learning method*: learning a policy that will **gives us a probability distribution over actions**.

##### Stable-Baselines3 setup steps:
1. You **create your environment** (in our case it was done above)
2. You define the **model you want to use and instantiate this model** `model = PPO("MlpPolicy")`
3. You **train the agent** with `model.learn` and define the number of training timesteps


**Sample Code:**

```
# Create environment
env = gym.make('LunarLander-v2')

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5))
```

------

```python
import inspect
import pandas as pd
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
```

------

**Inspect default PPO arguments**
```python
args = inspect.getfullargspec(PPO).args
defaults = inspect.getfullargspec(PPO).defaults
defaults = [None]*(len(args)-len(defaults)) + list(defaults)
annotations = inspect.getfullargspec(PPO).annotations.values()
annotations = [None]*(len(args)-len(annotations)) + list(annotations)
ppo_default_args = {arg:[default, annotation] for arg,default,annotation in zip(args, defaults, annotations)}
pd.DataFrame(ppo_default_args, index=["Default Value", "Annotation"]).T
```
<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Default Value</th>
      <th>Annotation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>self</th>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>policy</th>
      <td>None</td>
      <td>typing.Union[str, typing.Type[stable_baselines3.common.policies.ActorCriticPolicy]]</td>
    </tr>
    <tr>
      <th>env</th>
      <td>None</td>
      <td>typing.Union[gym.core.Env, stable_baselines3.common.vec_env.base_vec_env.VecEnv, str]</td>
    </tr>
    <tr>
      <th>learning_rate</th>
      <td>0.0003</td>
      <td>typing.Union[float, typing.Callable[[float], float]]</td>
    </tr>
    <tr>
      <th>n_steps</th>
      <td>2048</td>
      <td>&lt;class 'int'&gt;</td>
    </tr>
    <tr>
      <th>batch_size</th>
      <td>64</td>
      <td>&lt;class 'int'&gt;</td>
    </tr>
    <tr>
      <th>n_epochs</th>
      <td>10</td>
      <td>&lt;class 'int'&gt;</td>
    </tr>
    <tr>
      <th>gamma</th>
      <td>0.99</td>
      <td>&lt;class 'float'&gt;</td>
    </tr>
    <tr>
      <th>gae_lambda</th>
      <td>0.95</td>
      <td>&lt;class 'float'&gt;</td>
    </tr>
    <tr>
      <th>clip_range</th>
      <td>0.2</td>
      <td>typing.Union[float, typing.Callable[[float], float]]</td>
    </tr>
    <tr>
      <th>clip_range_vf</th>
      <td>None</td>
      <td>typing.Union[NoneType, float, typing.Callable[[float], float]]</td>
    </tr>
    <tr>
      <th>normalize_advantage</th>
      <td>True</td>
      <td>&lt;class 'bool'&gt;</td>
    </tr>
    <tr>
      <th>ent_coef</th>
      <td>0.0</td>
      <td>&lt;class 'float'&gt;</td>
    </tr>
    <tr>
      <th>vf_coef</th>
      <td>0.5</td>
      <td>&lt;class 'float'&gt;</td>
    </tr>
    <tr>
      <th>max_grad_norm</th>
      <td>0.5</td>
      <td>&lt;class 'float'&gt;</td>
    </tr>
    <tr>
      <th>use_sde</th>
      <td>False</td>
      <td>&lt;class 'bool'&gt;</td>
    </tr>
    <tr>
      <th>sde_sample_freq</th>
      <td>-1</td>
      <td>&lt;class 'int'&gt;</td>
    </tr>
    <tr>
      <th>target_kl</th>
      <td>None</td>
      <td>typing.Optional[float]</td>
    </tr>
    <tr>
      <th>tensorboard_log</th>
      <td>None</td>
      <td>typing.Optional[str]</td>
    </tr>
    <tr>
      <th>create_eval_env</th>
      <td>False</td>
      <td>&lt;class 'bool'&gt;</td>
    </tr>
    <tr>
      <th>policy_kwargs</th>
      <td>None</td>
      <td>typing.Optional[typing.Dict[str, typing.Any]]</td>
    </tr>
    <tr>
      <th>verbose</th>
      <td>0</td>
      <td>&lt;class 'int'&gt;</td>
    </tr>
    <tr>
      <th>seed</th>
      <td>None</td>
      <td>typing.Optional[int]</td>
    </tr>
    <tr>
      <th>device</th>
      <td>auto</td>
      <td>typing.Union[torch.device, str]</td>
    </tr>
    <tr>
      <th>_init_setup_model</th>
      <td>True</td>
      <td>&lt;class 'bool'&gt;</td>
    </tr>
  </tbody>
</table>
</div>

------

**Define a PPO MlpPolicy architecture**

```python
model = PPO("MlpPolicy", env, verbose=1)
```
```text
    Using cuda device
```

**Note:** 
* We use a Multilayer Perceptron because the observations are vectors instead of images.

* Recommended Values:

| Argument   | Value |
| ---------- | ----- |
| n_steps    | 1024  |
| batch_size | 64    |
| n_epochs   | 4     |
| gamma      | 0.999 |
| gae_lambda | 0.98  |
| ent_coef   | 0.01  |
| verbose    | 1     |

------


### Train the PPO agent

**Train the model**
```python
model.learn(total_timesteps=int(2000000))
```
```text
    ---------------------------------
    | rollout/           |          |
    |    ep_len_mean     | 94.8     |
    |    ep_rew_mean     | -199     |
    | time/              |          |
    |    fps             | 2891     |
    |    iterations      | 1        |
    |    time_elapsed    | 11       |
    |    total_timesteps | 32768    |
    ---------------------------------
...
    ------------------------------------------
    | rollout/                |              |
    |    ep_len_mean          | 187          |
    |    ep_rew_mean          | 281          |
    | time/                   |              |
    |    fps                  | 593          |
    |    iterations           | 62           |
    |    time_elapsed         | 3421         |
    |    total_timesteps      | 2031616      |
    | train/                  |              |
    |    approx_kl            | 0.0047587324 |
    |    clip_fraction        | 0.0585       |
    |    clip_range           | 0.2          |
    |    entropy_loss         | -0.469       |
    |    explained_variance   | 0.986        |
    |    learning_rate        | 0.0003       |
    |    loss                 | 3.62         |
    |    n_updates            | 610          |
    |    policy_gradient_loss | -0.0007      |
    |    value_loss           | 11.5         |
    ------------------------------------------

    <stable_baselines3.ppo.ppo.PPO at 0x7fcc807b8410>
```

------


### Evaluate the agent

- We can evaluate the model's performance using the [`evaluate_policy()`](https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html#stable_baselines3.common.evaluation.evaluate_policy) method.
- [Example](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#basic-usage-training-saving-loading)

**Create a new environment for evaluation**
```python
eval_env = gym.make('LunarLander-v2')
```

------

**Evaluate the model with 10 evaluation episodes and deterministic=True**
```python
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
```

------

**Print the results**
```python
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
```
```text
    mean_reward=78.15 +/- 94.84891574522395
```

------


### Publish our trained model on the Hub

* We can use the `package_to_hub()` method to evaluate the model, record a replay, generate a model card, and push the model to the Hub in a single line of code.
* [**Leaderboard**](https://huggingface.co/spaces/ThomasSimonini/Lunar-Lander-Leaderboard)
* The `package_to_hub()` method returns a link to a Hub model repository such as https://huggingface.co/osanseviero/test_sb3. 
* Model repository features:
    * A video preview of your agent at the right. 
    * Click "Files and versions" to see all the files in the repository.
    * Click "Use in stable-baselines3" to get a code snippet that shows how to load the model.
    * A model card (`README.md` file) which gives a description of the model
* Hugging Face Hub uses git-based repositories so we can update the model with new versions.

Connect to Hugging Face Hub:
1. Create Hugging Face account https://huggingface.co/join
2. Create a new authentication token (https://huggingface.co/settings/tokens) **with write role**
3. Run the `notebook_login()` method.


**Log into Hugging Face account**
```python
notebook_login()
!git config --global credential.helper store
```
```text
    Login successful
    Your token has been saved to /root/.huggingface/token
```

------

`package_to_hub` function arguments:
- `model`: our trained model.
- `model_name`: the name of the trained model that we defined in `model_save`
- `model_architecture`: the model architecture we used (e.g., PPO)
- `env_id`: the name of the environment, in our case `LunarLander-v2`
- `eval_env`: the evaluation environment defined in eval_env
- `repo_id`: the name of the Hugging Face Hub Repository that will be created/updated `(repo_id = {username}/{repo_name})`
    * **Example format:** {username}/{model_architecture}-{env_id}
- `commit_message`: message of the commit

------

```python
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_sb3 import package_to_hub
```

------

**Push the model to the Hugging Face Hub**
```python
# Define the name of the environment
env_id = "LunarLander-v2"

# Create the evaluation env
eval_env = DummyVecEnv([lambda: gym.make(env_id)])

# Define the model architecture we used
model_architecture = "ppo"

## Define a repo_id
## repo_id is the id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
repo_id = f"cj-mills/{model_architecture}-{env_id}"

model_name = f"{model_architecture}-{env_id}"

## Define the commit message
commit_message = f"Upload {model_name} model with longer training session"

# method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
package_to_hub(model=model, # Our trained model
               model_name=model_name, # The name of our trained model 
               model_architecture=model_architecture, # The model architecture we used: in our case PPO
               env_id=env_id, # Name of the environment
               eval_env=eval_env, # Evaluation Environment
               repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
               commit_message=commit_message)

```
```text
    'https://huggingface.co/cj-mills/ppo-LunarLander-v2'
```

------



### Some additional challenges

* Train for more steps.
* Try different [hyperparameters](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters) of `PPO`. 
* Check the [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) and try another model such as DQN.
* Try using the [CartPole-v1](https://www.gymlibrary.ml/environments/classic_control/cart_pole/), [MountainCar-v0](https://www.gymlibrary.ml/environments/classic_control/mountain_car/) or [CarRacing-v0](https://www.gymlibrary.ml/environments/box2d/car_racing/) environments.




## References

* [The Hugging Face Deep Reinforcement Learning Class](https://github.com/huggingface/deep-rl-class)
* [An Introduction to Deep Reinforcement Learning](https://huggingface.co/blog/deep-rl-intro)





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->