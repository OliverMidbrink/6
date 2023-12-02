1. install miniconda
2. conda create --name 6 python=3.10
3. Install latest nvidia drivers
4. install latest tested cuda configuration (https://www.tensorflow.org/install/source#gpu)
5. conda activate 6
6. Run these:

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# Add commands to the scripts
printf 'export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}\nexport LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib/\n' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
printf 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}\nunset OLD_LD_LIBRARY_PATH\n' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# Run the script once
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install --upgrade pip

pip install tensorflow==2.11
(or latest version you saw in the link (latest compatible build structure)

8. Test installation by running python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
If you see an empty array it did not work. If you see an array that contains GPU info. It worked. Congrats!

Cheers and have a nice day :)

New software on https://github.com/Molytica/M
