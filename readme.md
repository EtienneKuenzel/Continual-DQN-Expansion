
# CAP-Net #


- Code for Continual-DQN-Expansion 

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
python multi_agent_training.py --curriculum="custom" --policy="CDE" --hidden_size=1024 --layer_count=2

### Evaluation ###

Parser needs to be added
