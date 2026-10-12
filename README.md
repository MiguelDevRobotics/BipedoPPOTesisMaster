<div align="center">

# Bipedal Locomotion with NVIDIA Isaac Lab

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3.2-blue.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Refer%20to%20Repo-green.svg)](LICENSE)

*A reinforcement learning framework for bipedal robot locomotion leveraging massive parallel simulation in NVIDIA Isaac Sim and Isaac Lab.*

</div>
---

# Bipedal Locomotion with Isaac Lab

This repository contains a bipedal robot simulation and reinforcement learning training project developed using NVIDIA Isaac Lab and Isaac Sim.

The project focuses on training a bipedal robot to achieve locomotion through deep reinforcement learning, using parallel simulation environments to accelerate the training process.

This work originated as part of a **Master's thesis in Robotics** and was further developed into related research on deep reinforcement learning for bipedal locomotion.

## Project Structure

```text
├── biped_USD/         # USD model files for the bipedal robot
├── bipedo/            # Isaac Lab project files, env configs, and scripts
│   ├── scripts/       # Training, evaluation, and inference scripts
│   └── ...
├── bipedo_mike.py     # Robot asset configuration file for Isaac Lab
└── policy.pt          # Pretrained reinforcement learning policy checkpoint
```

## Software

| Component | Version |
| :--- | :--- |
| **Operating System** | Ubuntu 22.04 LTS (Jammy Jellyfish) |
| **NVIDIA Isaac Sim** | 5.1.0 *(Recommended)* |
| **NVIDIA Isaac Lab** | 2.3.2 *(Recommended)* |
| **PyTorch** | 2.7.0+cu128 |
| **PyTorch CUDA Runtime** | 12.8 |
| **NVIDIA Driver** | 580.178.04 |
| **RL Framework** | RSL-RL |
| **Algorithm** | Proximal Policy Optimization (PPO) |

> **Important:** Using the exact software versions specified above is strongly recommended to reproduce the original experimental results. Different versions may introduce breaking changes in robot configurations or environment definitions.

For installation instructions, refer to the
[Isaac Sim 5.1.0 Documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/).
[Isaac Lab 2.3.2 Documentation](https://isaac-sim.github.io/IsaacLab/v2.3.2/index.html).

### Hardware Configuration

The training experiments were performed using the following hardware:

| Component                        | Specification                      |
| -------------------------------- | ---------------------------------- |
| GPU                              | NVIDIA GeForce RTX 4080 Laptop GPU |
| GPU Memory                       | 12 GB VRAM                         |
| System Memory                    | 64 GB DDR5 RAM                     |

This configuration describes the environment used for the original
experiments.

## Setup

### 1. Configure the USD Model Path

The first step is to configure the absolute path to the bipedal robot USD model.

Open `bipedo_mike.py` and locate the following line:

``` python

usd_path=f"/absolute/path/to/yout/usd/bipedo.usd"

```

Replace the placeholder with the absolute path to the USD file located inside the `biped_USD` directory.

For example:

``` python

usd_path="/home/username/path/to/project/biped_USD/bipedo.usd"

```

Replace `/home/username/path/to/project/` with the actual absolute path to your project directory.

**Important:** Make sure the path points directly to `bipedo.usd` and that the file exists at that location.

### 2. Integrate the Robot Configuration into Isaac Lab

Copy `bipedo_mike.py` into the robot configuration directory of your Isaac Lab installation:

``` bash

IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/

```

For example, if Isaac Lab is installed at `/home/username/IsaacLab`, the destination will be:

``` bash

/home/username/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/bipedo_mike.py

```

This directory contains configuration files for other robots supported by Isaac Lab.

After copying the file, the robot configuration will be available within the Isaac Lab assets package for use by the project.

### 3. Verify the Project Configuration

Before starting a training session, verify that:

- The USD path in `bipedo_mike.py` points to the correct file.
- The robot configuration file has been copied into the Isaac Lab robots directory.
- Isaac Lab is installed and its Python environment is activated.
- The project environment and task configuration are available.

Once these steps are complete, you can start training or evaluating the bipedal robot.

## Reinforcement Learning Configuration

The bipedal robot is trained using Proximal Policy Optimization (PPO),
a reinforcement learning algorithm commonly used for continuous
control tasks.

The PPO configuration is defined in the following file:

[`rsl_rl_ppo_cfg.py`](bipedo/source/bipedo/bipedo/tasks/manager_based/bipedo/agents/rsl_rl_ppo_cfg.py)

This file contains the training configuration used by the RSL-RL
reinforcement learning framework, including the algorithm parameters
and other settings required for policy optimization.

To maintain consistency between the documentation and the
implementation, refer to the configuration file in the repository
for the exact PPO parameter values.

### Parallel Training

The training experiments were configured to use 4,000 parallel
simulation environments.

Parallel simulation allows the reinforcement learning agent to
collect experience from multiple instances of the robot simultaneously,
increasing the amount of experience collected per simulation step.

The number of environments that can be executed efficiently depends
on the available GPU memory and computational resources.

## Training

To train the bipedal robot using reinforcement learning, activate your Isaac Lab Python environment and execute the following command from the directory containing the `bipedo` folder:

``` bash

python bipedo/scripts/rsl_rl/train.py --task=Template-Bipedo-v0 --num_envs=4000 --headless

```

### Training Parameters

| Parameter | Description |
|---|---|
| `--task=Template-Bipedo-v0` | Specifies the bipedal robot training task. |
| `--num_envs=4000` | Runs 4,000 simulation environments in parallel. |
| `--headless` | Runs the simulation without displaying the graphical interface. |

Using multiple simulation environments allows the reinforcement learning algorithm to collect experience from many parallel simulations, potentially accelerating the learning process.

**Note:** The required GPU memory and computational resources depend on the hardware and simulation configuration. If your GPU cannot handle 4,000 environments, try reducing `--num_envs`.

For example:

``` bash

python bipedo/scripts/rsl_rl/train.py --task=Template-Bipedo-v0 --num_envs=2000 --headless

```

## Running a Trained Policy

Once a trained policy is available, you can run the evaluation or playback script to observe the robot's behavior.

Execute:

``` bash

python bipedo/scripts/rsl_rl/play.py --task=Template-Bipedo-v0 --num_envs=500

```

### Playback Parameters

| Parameter | Description |
|---|---|
| `--task=Template-Bipedo-v0` | Loads the bipedal robot task. |
| `--num_envs=500` | Specifies 500 simulation environments for playback. |

Unlike the training command, this example does not include `--headless`, allowing the simulation to display a graphical interface.

The playback script must be able to locate a compatible trained checkpoint. If no checkpoint is available, run the training script first and ensure the playback configuration points to the resulting model.

## Keyboard Control and Policy Inference

This repository includes a keyboard-based inference script that allows
users to control the bipedal robot using a pretrained reinforcement
learning policy.

The pretrained policy `policy.pt` is included in the root directory
of this repository. No additional checkpoint download is required.

### Pretrained Policy

The training implementation uses Proximal Policy Optimization (PPO)
through the RSL-RL framework. The corresponding PPO configuration is
available at:

[`rsl_rl_ppo_cfg.py`](bipedo/source/bipedo/bipedo/tasks/manager_based/bipedo/agents/rsl_rl_ppo_cfg.py)

### 1. Configure the Policy Path

Open `bipedo/scripts/inferencia.py` and locate the following line:

``` python

policy_path = os.path.abspath("/path/to/policy.pt")

```

Replace the placeholder with the absolute path to the `policy.pt` file
in the root directory of the cloned repository.

For example:

``` python

policy_path = os.path.abspath("/home/username/path/to/project/policy.pt")

```

Replace `/home/username/path/to/project/` with the actual absolute path
to your project directory.

### 2. Run the Inference Script

Activate your Isaac Lab Python environment and execute the following
command from the root directory of the repository:

``` bash

python bipedo/scripts/inferencia.py

```

The script loads the pretrained policy and launches the robot simulation.

### 3. Keyboard Controls

Use the arrow keys to control the bipedal robot during simulation:

| Key | Control |
|---|---|
| ↑ Up Arrow | Forward |
| ↓ Down Arrow | Backward |
| ← Left Arrow | Turn left |
| → Right Arrow | Turn right |

The keyboard inputs provide movement commands to the trained policy,
which generates the corresponding robot actions.

This feature allows users to interactively evaluate the learned
locomotion policy without running a new training session.

### Notes

- Make sure the USD model path in `bipedo_mike.py` is configured correctly.
- Ensure that `policy.pt` is accessible at the path specified in
  `inferencia.py`.

## Optional Rough Terrain Generation

The repository includes an optional rough terrain generator implemented
using the terrain generation utilities provided by NVIDIA Isaac Lab.

This feature was used to evaluate the bipedal robot's locomotion on
uneven surfaces. It is disabled in the default environment configuration.

### Terrain Configuration

The terrain generator is defined in:

[`bipedo_env_cfg.py`](bipedo/source/bipedo/bipedo/tasks/manager_based/bipedo/bipedo_env_cfg.py)

The configuration uses `TerrainGeneratorCfg` and
`HfRandomUniformTerrainCfg` to generate terrain with random height
variations.

### Enabling Rough Terrain

By default, the environment uses a flat plane. In `bipedo_env_cfg.py`,
locate the following configuration:

```python

self.scene.terrain.terrain_generator = None 
self.scene.terrain.terrain_type = "plane"

```

To enable rough terrain generation, replace it with:

```python

self.scene.terrain.terrain_generator = ROUGH_TERRAIN_CFG 
self.scene.terrain.terrain_type = "generator"

```

After making these changes, run the simulation or inference script
to visualize the generated terrain.

### Evaluation on Rough Terrain

The pretrained policy `policy.pt` was trained using the original
repository configuration with 4,000 parallel simulation environments.

Although the policy was not specifically trained using the rough
terrain configuration, the robot was able to perform several walking
steps on uneven surfaces.

A quantitative evaluation of the robot's locomotion performance is
presented in the related scientific publication.

For the experimental methodology and detailed results, refer to:

**Deep Reinforcement Learning for Bipedal Locomotion Using PPO and
Parallel Training in Isaac Sim**

- [Read the article on Springer](https://link.springer.com/chapter/10.1007/978-3-032-37444-8_44)
- [DOI: 10.1007/978-3-032-37444-8_44](https://doi.org/10.1007/978-3-032-37444-8_44)
  
## Research Background

This project was originally developed as part of a **Master's thesis in Robotics**. The research focuses on bipedal locomotion, robot simulation, and reinforcement learning using NVIDIA Isaac Sim and Isaac Lab.

The project provides a practical starting point for exploring reinforcement learning-based locomotion, parallel simulation, and the development of control policies for bipedal robots.

The research was also associated with a scientific publication on deep reinforcement learning for bipedal locomotion using Proximal Policy Optimization (PPO) and parallel training in Isaac Sim.

### Thesis

The original Master's thesis is available through the institutional repository:

- [Read the Master's thesis](http://repositorio.utm.mx:8080/jspui/handle/123456789/548)

### Videos

Videos demonstrating the project and its results are available on YouTube:

- [Watch the project videos](https://youtube.com/playlist?list=PLQBIW6DdZczY&si=8E8SDywnBcPs8h3c)

### Related Scientific Publication

**Deep Reinforcement Learning for Bipedal Locomotion Using PPO and Parallel Training in Isaac Sim**

Authors: Miguel-de-Jesús Santiago-Vásquez, Oscar-D. Ramírez-Cardenas, and José-A. Arias.

Published by Springer as part of *Artificial Intelligence – COMIA 2026*.

- [Read the article on Springer](https://link.springer.com/chapter/10.1007/978-3-032-37444-8_44)
- [DOI: 10.1007/978-3-032-37444-8_44](https://doi.org/10.1007/978-3-032-37444-8_44)

## Acknowledgments

This project is the result of research conducted as part of a Master's degree in Robotics. It aims to contribute to the development and evaluation of reinforcement learning techniques for bipedal robotic locomotion using physics-based simulation.

## License

This project is licensed under the Apache License 2.0.
You may obtain a copy of the license in the [LICENSE](LICENSE) file.

   Copyright 2026 Miguel de Jesus Santiago Vasquez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
