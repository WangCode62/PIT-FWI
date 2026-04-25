# PIT-FWI
In this repository, I implemented the physics-informed Transformer for full-waveform inversion(PIT-FWI). Using this method, we realized its application in the VSP acquisition geometry. The overall architecture of the proposed method is illustrated in the following figure.
<div align="center">
  <img src="https://github.com/user-attachments/assets/f5f83943-51b6-4d27-9187-af219c90462f" width="700">
</div>
## Installation

To use this code, you first need to set up the environment. The required dependencies can be installed using the provided [environment.yml](./environment.yml) file.

You can also download it directly:  
[Download environment.yml](https://github.com/WangCode62/PIT-FWI/raw/main/environment.yml)

Then run:
```bash
conda env create -f environment.yml
conda activate your_env_name
```

### Scripts for Running FWI
***In this repo, there are four scripts for running FWI:***  

1.[main.ipynb](./main.ipynb)The main scripts demonstrate the performance of PIT-FWI.  
2.[forward_module.py](./forward_module.py)provides the forward modeling workflow.  
3.[functions_module.py](./functions_module.py)provides the functions used in the main script.  
4.[network_module.py](./network_module.py)provides the neural network architecture.  
5.[plot_module.py](./plot_module.py)provides visualization and plotting utilities.  

The result of running this code for 21 shots with 1000 epochs on the Sigsbee model is shown in the following figures.
