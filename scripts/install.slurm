ROBOSUITE=true

if [ ROBOSUITE ]; then
  env_name="robosuite"
else
  env_name="robocasa"
fi

conda env create --prefix ./envs/$env_name -f environment.yml
conda activate envs/$env_name

if [ ROBOSUITE ]; then
  # Robosuite
  pip install -e ../cloned/mimicgen
  pip install pynput==1.6.0
  pip install -e ../cloned/robosuite
  pip install -e ../cloned/robomimic
  pip install mujoco==2.3.2
  pip install numba==0.58.1
else
  # Robocasa
  #pip install -e ../cloned/robocasa/robosuite_casa/robosuite/
  #pip install -e ../cloned/robocasa/
  #pip install -e ../cloned/robocasa/mimicgen_casa/
  #pip install -e ../cloned/robocasa/robomimic_casa/
fi

git clone git@github.com:facebookresearch/r3m.git
pip install -e ./r3m
pip install --upgrade protobuf
