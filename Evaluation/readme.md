# Evaluation #

## Evaluation Done ##

|     | DQN | PAU | | EWC0.1| 0.5| 1 | -0.5| |CDE0.1 | 0.5| 1 | -0.5| 
|-----|-----|-----|-|-------|----|---|-----|-|-------|----|---|-----|
| PMD | x   | x   | | x     | x  | x |  x  | |       |  x |   |     |
| PDM | x   |     | |       | x  |   |     | |       |    |   |     |
| DPM |     |     | |       | x  |   |     | |       |    |   |     |
| DMP |     |     | |       | x  |   |     | |       |    |   |     |
| MPD |     |     | |       | x  |   |     | |       |    |   |     |
| MDP |     |     | |       | x  |   |     | |       |    |   |     |

## Sub-Environment Performance ##
Score and Completions Evaluation of the Pathfinding, Malfunction, Deadlock and Evaluation Environment during the training.
All Evaluations were made with a 2 Layers à 1024 Neurons
Sub-Environments have their first letter as Abreviations D=Deadlock, M=Malfunction, P=Pathfinding.

### DQN ###
#### Curriculum PMD ####
<p float="left">
  <img src="images\subenv\eval_DQN-2x1024_customPMD_completions.png" width="49%" />
  <img src="images\subenv\eval_DQN-2x1024_customPMD_score.png" width="49%" />
</p> 

#### Curriculum PDM ####
<p float="left">
  <img src="images\subenv\eval_DQN-2x1024_customPDM_completions.png" width="49%" />
  <img src="images\subenv\eval_DQN-2x1024_customPDM_score.png" width="49%" />
</p> 


### PAU (m=5, n=4) ###
#### Curriculum PMD ####
<p float="left">
  <img src="images\subenv\eval_PAU-2x1024_customPMD_completions.png" width="49%" />
  <img src="images\subenv\eval_PAU-2x1024_customPMD_score.png" width="49%" />
</p>


### CDE (lambda=0.5) (m=5 n=4)###
Evaluation for CDE just for one Sub-Network so just partially represantive
#### Curriculum PMD ####
<p float="left">
  <img src="images\subenv\eval_CDE0.5-2x1024_customPMD_completions.png" width="49%" />
  <img src="images\subenv\eval_CDE0.5-2x1024_customPMD_score.png" width="49%" />
</p>

### EWC (lambda=-0.5) ###
#### Curriculum PMD ####
<p float="left">
  <img src="images\subenv\eval_EWC-0.5-2x1024_customPMD_completions.png" width="49%" />
  <img src="images\subenv\eval_EWC-0.5-2x1024_customPMD_score.png" width="49%" />
</p>

### EWC (lambda=0.1) ###
#### Curriculum PMD ####
<p float="left">
  <img src="images\subenv\eval_EWC0.1-2x1024_customPMD_completions.png" width="49%" />
  <img src="images\subenv\eval_EWC0.1-2x1024_customPMD_score.png" width="49%" />
</p>

### EWC (lambda=0.5) ###
#### Curriculum PMD ####
<p float="left">
  <img src="images\subenv\eval_EWC0.5-2x1024_customPMD_completions.png" width="49%" />
  <img src="images\subenv\eval_EWC0.5-2x1024_customPMD_score.png" width="49%" />
</p>

#### Curriculum PDM ####
<p float="left">
  <img src="images\subenv\eval_EWC0.5-2x1024_customPDM_completions.png" width="49%" />
  <img src="images\subenv\eval_EWC0.5-2x1024_customPDM_score.png" width="49%" />
</p>

#### Curriculum MPD ####
<p float="left">
  <img src="images\subenv\eval_EWC0.5-2x1024_customMPD_completions.png" width="49%" />
  <img src="images\subenv\eval_EWC0.5-2x1024_customMPD_score.png" width="49%" />
</p>

#### Curriculum MDP ####
<p float="left">
  <img src="images\subenv\eval_EWC0.5-2x1024_customMDP_completions.png" width="49%" />
  <img src="images\subenv\eval_EWC0.5-2x1024_customMDP_score.png" width="49%" />
</p>

#### Curriculum DMP ####
<p float="left">
  <img src="images\subenv\eval_EWC0.5-2x1024_customDMP_completions.png" width="49%" />
  <img src="images\subenv\eval_EWC0.5-2x1024_customDMP_score.png" width="49%" />
</p>

#### Curriculum DPM ####
<p float="left">
  <img src="images\subenv\eval_EWC0.5-2x1024_customDPM_completions.png" width="49%" />
  <img src="images\subenv\eval_EWC0.5-2x1024_customDPM_score.png" width="49%" />
</p>



### EWC (lambda=1) ###
#### Curriculum PMD ####
<p float="left">
  <img src="images\subenv\eval_EWC1-2x1024_customPMD_completions.png" width="49%" />
  <img src="images\subenv\eval_EWC1-2x1024_customPMD_score.png" width="49%" />
</p>