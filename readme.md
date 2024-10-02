
# Mitigating the Stability-Plasticity Dilemma in Adaptive Train Scheduling with Curriculum-Driven Continual DQN Expansion #
This repository contains the code and resources for our paper, which addresses the stability-plasticity dilemma in adaptive train scheduling. In this work, we introduce different curricula for train scheduling and an algorithm called "Continual Deep Q-Network Expansion" (CDE) to improve agent adaptability in non-stationary environments. Our method dynamically adjusts Q-function subspaces and utilizes Elastic Weight Consolidation (EWC) and Rational Padé Activation Function to adress Catastrophic Forgetting and decreasing Network Plasticity, achieving superior learning efficiency and generalization compared to traditional RL baselines. The curriculum design emphasizes skill building through adjacent tasks, helping agents retain learned knowledge while effectively acquiring new behaviors.

- Code for the [Continual-DQN-Expansion-Algorithm](MARL/flatland-starter-kit-master/baselines/reinforcement_learning/dddqn_policy.py)
- Code Implementations of 4 different [Base Learning Curriculum](MARL/flatland-starter-kit-master/baselines/multi_agent_training.py)
- Code to evaluate the [Results](Evaluation)
- Link to the [Paper](https://arxiv.org/pdf/2408.09838)
- Simulator used is [Flatland-RL](https://github.com/flatland-association/flatland-rl) by AICrowd

## Continual-DQN-Expansion (CDE) ##
<p float="left">
  <img src="Evaluation\images\cde.jpg" width="100%" />
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

### DQN Evaluation ###
<p float="left">
  <img src="Evaluation\images\DQN_Completions.png" width="49%" />
  <img src="Evaluation\images\DQN_Score.png" width="49%" />
</p>
