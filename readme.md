
# Continual-DQN-Expansion #
- Code for Continual-DQN-Expansion 
- Code to evaluate Results
## Simulator ##

- Simulator used is Flatland-RL : https://github.com/flatland-association/flatland-rl

## Usage ##
### Requirements ###
- You can create an environment using  ```conda env create -f environment.yml```


### Training ###
To test if Install is working correctly:
```
python multi_agent_training.py --curriculum="test" 
```
Run CDE on custom curriculum:
```
python multi_agent_training.py --curriculum="custom" --policy="CDE" --hidden_size=1024 --layer_count=2
```

### Evaluation ###
Create Animation of PAU-Activation-Function 
```
python eval_weights.py --file="weights8x256.csv" --network="0" --layer=0
```

Create Plot of Trainingand Evaluation-Completions/Score
```
python eval_training.py --file="score2x1024.csv" --type="score"
```
```
python eval_training.py --file="completions2x1024.csv" --type="completions"
```
## Results ##
| Completions                                                   | Scores                                          |
| ----------------------------------------------------- | --------------------------------------------- |
| ![test](Evaluation\images\completions-comparison-layer size.png) | ![test](Evaluation\images\score-comparison-layer size.png) |

<img title="Test" alt="Alt text" src="Evaluation\images\completions-comparison-layer-size.png">