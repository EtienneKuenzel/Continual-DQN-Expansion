# Evaluation #

## Evaluation ##
### Completion Rate ###
|     | DQN               | PAU              | | EWC0.1            | 0.5                | 1                 | -0.5              | |CDE0.1 | 0.5| 1                   | -0.5| 
|-----|-------------------|------------------|-|-------------------|--------------------|-------------------|-------------------|-|-------|----|---------------------|-----|
| PMD |0.10998578535891973|0.1311846689895474| |0.24004000666777675|0.18346080796673922 |0.14735930735930755|0.10354029532111728| |       |  0.25259515570934093 | 0.14408113306522116 |     |
| PDM |0.06465595087592581|                  | |                   |0.09500177242112744 |   |     | |       |    |   |     |
| DPM |0.05660036166365291|                  | |                   |0.06928034371643421 |   |     | |       |    |   |     |
| DMP |                   |                  | |                   |0.11357586512866051 |   |     | |       |    |   |     |
| MPD |                   |                  | |                   |0.2170003393281295  |   |     | |       |    |   |     |
| MDP |                   |0.0804020100502516| |                   |0.09813664596273322 |   |     | |       |    |   |     |

### Score ###
|     | DQN | PAU | | EWC0.1| 0.5| 1 | -0.5| |CDE0.1 | 0.5| 1 | -0.5| 
|-----|-----|-----|-|-------|----|---|-----|-|-------|----|---|-----|
| PMD | x   | x   | | x     | x  | x |  x  | |       |  x | x |     |
| PDM | x   |     | |       | x  |   |     | |       |    |   |     |
| DPM | x   |     | |       | x  |   |     | |       |    |   |     |
| DMP |     |     | |       | x  |   |     | |       |    |   |     |
| MPD |     |     | |       | x  |   |     | |       |    |   |     |
| MDP |     |  x  | |       | x  |   |     | |       |    |   |     |

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

#### Curriculum DPM ####
<p float="left">
  <img src="images\subenv\eval_DQN-2x1024_customDPM_completions.png" width="49%" />
  <img src="images\subenv\eval_DQN-2x1024_customDPM_score.png" width="49%" />
</p> 

### PAU (m=5, n=4) ###
#### Curriculum PMD ####
<p float="left">
  <img src="images\subenv\eval_PAU-2x1024_customPMD_completions.png" width="49%" />
  <img src="images\subenv\eval_PAU-2x1024_customPMD_score.png" width="49%" />
</p>

#### Curriculum MDP ####
<p float="left">
  <img src="images\subenv\eval_PAU-2x1024_customMDP_completions.png" width="49%" />
  <img src="images\subenv\eval_PAU-2x1024_customMDP_score.png" width="49%" />
</p>

### CDE (lambda=0.5) (m=5 n=4)###
Evaluation for CDE just for the best Sub-Network at the moment
#### Curriculum PMD ####
<p float="left">
  <img src="images\subenv\eval_CDE0.5-2x1024_customPMD_completions.png" width="49%" />
  <img src="images\subenv\eval_CDE0.5-2x1024_customPMD_score.png" width="49%" />
</p>

### CDE (lambda=1) (m=5 n=4)###
Evaluation for CDE just for the best Sub-Network at the moment
#### Curriculum PMD ####
<p float="left">
  <img src="images\subenv\eval_CDE1-2x1024_customPMD_completions.png" width="49%" />
  <img src="images\subenv\eval_CDE1-2x1024_customPMD_score.png" width="49%" />
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