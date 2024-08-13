
# Continual-DQN-Expansion (CDE)#
- Code for the Continual-DQN-Expansion-Algorithm
- Code Implementations of 4 different Learning Curriculum (+5 extra Curricula with changed task sequences)
- Code to evaluate the Results

## Simulator ##

- Simulator used is Flatland-RL : https://github.com/flatland-association/flatland-rl

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
