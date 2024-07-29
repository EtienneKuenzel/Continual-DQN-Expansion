# Evaluation #

## Evaluation ##
### Completion Rate ###
|     | DQN           | PAU           | | EWC0.1            | 0.5            | 1   | -0.5| |CDE0.1 | 0.5  | 1               | -0.5| 
|-----|---------------|---------------|-|-------------------|----------------|-----|-----|-|-------|------|-----------------|-----|
| PMD |0.110          |0.131          | |0.240              |0.183           |0.147|0.104| |0.18   |0.253 | 0.144 max(0.94) |     |
| PDM |0.065 max(0.21)|0.088          | |                   |0.095 max(0.21) |     |     | |       |0.104 |                 |     |
| DPM |0.057 max(0.11)|0.045          | |                   |0.069 max(0.18) |     |     | |       |0.113 |                 |     |
| DMP |0.086          |0.062          | |                   |0.114           |     |     | |       |0.101 |                 |     |
| MPD |0.037          |0.061          | |                   |0.217           |     |     | |       |0.189 |                 |     |
| MDP |0.084          |0.080 max(0.13)| |                   |0.098 max(0.18) |     |     | |       |0.143 |                 |     |

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

#### Curriculum DMP ####
<p float="left">
  <img src="images\subenv\eval_DQN-2x1024_customDMP_completions.png" width="49%" />
  <img src="images\subenv\eval_DQN-2x1024_customDMP_score.png" width="49%" />
</p> 

#### Curriculum MPD ####
<p float="left">
  <img src="images\subenv\eval_DQN-2x1024_customMPD_completions.png" width="49%" />
  <img src="images\subenv\eval_DQN-2x1024_customMPD_score.png" width="49%" />
</p> 

#### Curriculum MDP ####
<p float="left">
  <img src="images\subenv\eval_DQN-2x1024_customMDP_completions.png" width="49%" />
  <img src="images\subenv\eval_DQN-2x1024_customMDP_score.png" width="49%" />
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

### Completion Rate ###
|     | DQN               | PAU              | | EWC0.1            | 0.5                | 1                 | -0.5              | |CDE0.1             | 0.5                  | 1                   | -0.5| 
|-----|-------------------|------------------|-|-------------------|--------------------|-------------------|-------------------|-|-------------------|----------------------|---------------------|-----|
| PMD |0.10998578535891973|0.1311846689895474| |0.24004000666777675|0.18346080796673922 |0.14735930735930755|0.10354029532111728| |0.17513736263736218|0.25259515570934093   | 0.14408113306522116 |     |
| PDM |0.06465595087592581|0.0881724260776632| |                   |0.09500177242112744 |                   |                   | |                   |0.10410714285714319|    |   |     
| DPM |0.05660036166365291|0.0448357233617716| |                   |0.06928034371643421 |                   |                   | |                   |0.11273559978862083|    |   |    
| DMP |0.08592777085927789|0.0618444846292949| |                   |0.11357586512866051 |   |                                   | |                   |0.10062111801242253|    |   |    
| MPD |0.03725917848055245|0.0609911054637867| |                   |0.2170003393281295  |   |                                   | |                   |0.18932874354561102|    |   |    
| MDP |0.08427672955974871|0.0804020100502516| |                   |0.09813664596273322 |   |                                   | |                   |0.14285714285714285|    |   |    
