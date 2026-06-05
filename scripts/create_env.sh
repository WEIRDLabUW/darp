ROBOSUITE=true

if [ "$ROBOSUITE" = "true" ]; then
  env_name="robosuite"
else
  env_name="robocasa"
fi

echo "Creating environment: $env_name"

# Initialize conda for shell script usage
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda env create --prefix ./envs/$env_name -f environment.yml -y
conda activate ./envs/$env_name

# Clone dependencies
echo "Cloning/Updating dependencies in ./cloned..."
mkdir -p cloned

if [ "$ROBOSUITE" = "true" ]; then
    # Standalone dependencies (for robosuite environment)
    if [ ! -d "cloned/mimicgen" ]; then
        git clone https://github.com/NVlabs/mimicgen.git cloned/mimicgen
        git -C cloned/mimicgen checkout 70048f1d0ec3aa1ab3420fd573526ebeb10cbced
    fi

    if [ ! -d "cloned/robosuite" ]; then
        git clone https://github.com/ARISE-Initiative/robosuite.git cloned/robosuite
        git -C cloned/robosuite checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa
    fi

    if [ ! -d "cloned/robomimic" ]; then
        git clone https://github.com/ARISE-Initiative/robomimic.git cloned/robomimic
        git -C cloned/robomimic checkout d0b37cf214bd24fb590d182edb6384333f67b661
    fi
else
    # Robocasa setup (contains its own versions of robosuite, mimicgen, robomimic)
    if [ ! -d "cloned/robocasa" ]; then
        git clone https://github.com/robocasa/robocasa.git cloned/robocasa
        git -C cloned/robocasa checkout 0eae0634a61ad2be33962c9de7000a2dd1ee573f
    fi

    if [ ! -d "cloned/robocasa/robosuite_casa/robosuite" ]; then
        mkdir -p cloned/robocasa/robosuite_casa
        git clone https://github.com/ARISE-Initiative/robosuite cloned/robocasa/robosuite_casa/robosuite
        git -C cloned/robocasa/robosuite_casa/robosuite checkout cb25aae7cac84c10409b96827a6c8d5a21f48f3a
    fi

    if [ ! -d "cloned/robocasa/mimicgen_casa" ]; then
        git clone https://github.com/NVlabs/mimicgen cloned/robocasa/mimicgen_casa
        git -C cloned/robocasa/mimicgen_casa checkout 36ef5b744eeac684e3e5c2809f403177b8fc15e3
    fi

    if [ ! -d "cloned/robocasa/robomimic_casa" ]; then
        git clone https://github.com/ARISE-Initiative/robomimic cloned/robocasa/robomimic_casa
        git -C cloned/robocasa/robomimic_casa checkout 271a76c2d55c8b0f94d3d589f26fcae0d47f64a1
    fi
fi

# R3M (Common) - Using HTTPS to avoid OpenSSL/SSH conflicts
if [ ! -d "cloned/r3m" ]; then
    git clone https://github.com/facebookresearch/r3m.git cloned/r3m
    git -C cloned/r3m checkout b2334e726887fa0206962d7984c69c5fb09cceab
fi

echo "Installing packages for $env_name..."

if [ "$ROBOSUITE" = "true" ]; then
  pip install -e cloned/mimicgen
  pip install pynput==1.6.0
  pip install -e cloned/robosuite
  pip install -e cloned/robomimic
  pip install mujoco==2.3.2
  pip install numba==0.58.1
else
  pip install -e cloned/robocasa/robosuite_casa/robosuite/
  pip install -e cloned/robocasa/mimicgen_casa/
  pip install -e cloned/robocasa/robomimic_casa/
  pip install -e cloned/robocasa/
fi

pip install -e cloned/r3m
pip install --upgrade protobuf
pip install optuna line_profiler

# --- MuJoCo Automation ---
# Create conda activation hooks to handle rendering and LD_LIBRARY_PATH automatically
ACTIVATE_DIR="./envs/$env_name/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
cat <<EOF > "$ACTIVATE_DIR/env_vars.sh"
#!/bin/sh
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$HOME/.mujoco/mujoco210/bin"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
EOF

