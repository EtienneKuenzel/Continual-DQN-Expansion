
# Mitigating the Stability-Plasticity Dilemma in Adaptive Train Scheduling with Curriculum-Driven Continual DQN Expansion #
## Abstract ##
A continual learning agent builds on previous experiences to develop increasingly complex behaviors by adapting to non-stationary and dynamic environments while preserving previously acquired knowledge. However, scaling these systems presents significant challenges, particularly in balancing the preservation of previous policies with the adaptation of new ones to current environments. This balance, known as the stability-plasticity dilemma, is especially pronounced in complex multi-agent domains such as the train scheduling problem, where environmental and agent behaviors are constantly changing, and the search space is vast. In this work, we propose addressing these challenges in the train scheduling problem using curriculum learning. We design a curriculum with adjacent skills that build on each other to improve generalization performance. Introducing a curriculum with distinct tasks introduces non-stationarity, which we address by proposing a new algorithm: Continual Deep Q-Network (DQN) Expansion (CDE). Our approach dynamically generates and adjusts Q-function subspaces to handle environmental changes and task requirements. CDE mitigates catastrophic forgetting through EWC while ensuring high plasticity using adaptive rational  activation functions. Experimental results demonstrate significant improvements in learning efficiency and adaptability compared to RL baselines and other adapted methods for continual learning, highlighting the potential of our method in managing the stability-plasticity dilemma in the adaptive train scheduling setting.

- Code for the [Continual-DQN-Expansion-Algorithm](MARL/flatland-starter-kit-master/baselines/reinforcement_learning/dddqn_policy.py)
- Code Implementations of 4 different [Base Learning Curricula](MARL/flatland-starter-kit-master/baselines/multi_agent_training.py)
- Code to evaluate the [Results](Evaluation)
- Link to the [Paper](https://arxiv.org/pdf/2408.09838)
- Simulator used is [Flatland-RL](https://github.com/flatland-association/flatland-rl) by AICrowd

## Continual-DQN-Expansion (CDE) ##
<p float="left">
  <img src="Evaluation\images\cde.jpg" width="50%" />
</p>



## Usage ##
### Requirements ###
-After cloning you can create an environment using:  
```conda env create -f environment.yml```

```python -m pip install flatland-rl```

```python -m pip install torch ~```

### Training ###
Run CDE on the Custom Curriculum:
```
python multi_agent_training.py --curriculum="customPMD" --policy="CDE" --hidden_size=1024 --layer_count=2 --ewc_lambda=0.5
```

### Evaluation ###
-Create Plot of Training and Evaluation-Completions/Score
```
python eval_training.py --file="score_***.csv" --type="score"
```
```
python eval_training.py --file="completions_****.csv" --type="completions"
```
-Create Animation of PAU-Activation-Function 
```
python eval_weights.py --file="weights_***.csv" --network="0" --layer=0
```
## Results ##
### Baselines ###
<p float="left">
  <img src="Evaluation\images\Baselines_Curriculum_Completions.png" width="49%" />
  <img src="Evaluation\images\Baselines_Curriculum_Scores.png" width="49%" />
</p>

### DQN Evaluation on the Custom Curriculum without Rehearsal ###
<p float="left">
  <img src="Evaluation\images\DQN_Completions.png" width="49%" />
  <img src="Evaluation\images\DQN_Score.png" width="49%" />
</p>
