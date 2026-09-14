# CONTEST
This repository contains the official implementation of the paper:
> **Conformance-Aware Counterfactual Explanations for Prescriptive Process Analytics**  
> *Ngoc-Diem Le, Alessandro Padella, Massimiliano de Leoni*  
> [[Paper (PDF)](stay-tunned)] 

## Abstract
Prescriptive process analytics aims to provide actionable recommendations for process instances predicted to lead to unsatisfactory outcomes. However, many existing systems operate as black boxes: they do not explain why a recommendation was made, making it difficult for stakeholders to understand or contest its rationale. Several methods exist for explaining process predictions, but none explains recommendations about the next activity and the prescribed resource. In this paper, this problem is tackled by introducing a counterfactual-based framework that identifies the minimal changes to a trace that would alter the recommended activity–resource pair. Existing counterfactual generation methods typically assume static, fixed-length feature vectors and cannot capture the sequential, temporally dependent structure of process traces. Our approach addresses this issue by generating counterfactuals in a conformance-aware latent space, shaped by a differentiable structural regularizer that aligns generated traces with observed ones. The feasibility of candidate explanations is then validated by an independently trained model and by a resource–activity compatibility check. An empirical evaluation on multiple event logs, using mined Declare constraints as a plausibility measure, shows that our framework not only explains recommendations but also matches the plausibility of the closest baseline. Unlike that baseline, however, it relies on an implicit conformance mechanism that requires no Declare-constraint mining at explanation time, and is therefore free from the mining-threshold sensitivity that limits explicit conformance approaches.

## Framework
<img width="5135" height="2928" alt="framework (1)" src="https://github.com/user-attachments/assets/92272dea-1d10-4a8b-a4fa-ce87f786ba8a" />

## Installation
**Dependencies**

This implementation requires the following Python libraries:
```python
pip install pandas numpy 
```

## Usage
1. Preprocessing Event Logs
2. 
3. 
```python
python 
```

## License
This project is licensed under the MIT License.
