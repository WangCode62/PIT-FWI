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
<div align="center">
  <img src="https://github.com/user-attachments/assets/c5634211-31b4-4803-9072-760feb429ee5" width="700">
</div>
Two velocity single-trace profiles are shown in the figure.
<div align="center">
  <img src="https://github.com/user-attachments/assets/095ee1d4-60d2-44ef-9a3e-8349b5a17300" width="550">
</div>

***If you would like to explore more experimental results, please run the other main scripts.***  

1.[main_low5.ipynb](./main_low5.ipynb)Demonstrates the inversion results of PIT-FWI under the condition of missing low-frequency information below 5 Hz.      
2.[main_low8.ipynb](./main_low8.ipynb)Demonstrates the inversion results of PIT-FWI under the condition of missing low-frequency information below 8 Hz.
3.[main_noise10.ipynb](./main_noise10.ipynb)Demonstrates the inversion results of PIT-FWI under the condition of a data signal-to-noise ratio of 10 dB..   
4.[main_noise20.ipynb](./main_noise20.ipynb)Demonstrates the inversion results of PIT-FWI under the condition of a data signal-to-noise ratio of 20 dB..   
