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
