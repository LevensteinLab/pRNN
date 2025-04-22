# modified from https://github.com/jcornford/dann_rnns/blob/main/make_venv.sh
#
module --force purge
module load python/3.9


VENV_NAME='PredictiveReplay_39'
VENV_DIR=$HOME'/venvs/'$VENV_NAME

echo 'Building virtual env: '$VENV_NAME' in '$VENV_DIR

mkdir $VENV_DIR
# Add .gitignore to folder just in case it is in a repo
# Ignore everything in the directory apart from the gitignore file
echo "*" > $VENV_DIR/.gitignore
echo "!.gitignore" >> $VENV_DIR/.gitignore

virtualenv $VENV_DIR

source $VENV_DIR'/bin/activate'
pip install --upgrade pip
pip cache purge

# install python packages not provided by modules

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
pip install pandas==1.5.3


pip install matplotlib==3.5.3
pip install scipy==1.12.0
pip install numpy==1.22.4
pip install scikit-learn==1.4.0
pip install pynapple==0.6.1

pip install zipp==3.17.0
pip install pyparsing==3.1.1
pip install importlib_metadata==7.0.1
pip install ruamel.yaml==0.18.6

pip3 install gym==0.21.0 --no-binary :all:
pip install gymnasium==0.29.1
pip install ratinabox==1.15.2

#pip3 install gym-minigrid  #For now, use mine:
cd $HOME/minigrid
# git clone git@github.com:dlevenstein/gym-minigrid.git
# git clone git@github.com:Alxec/minigrid.git
cd minigrid
pip3 install -e .

# set up MILA jupyterlab
echo which ipython
#ipython kernel install --user --name=PredictiveReplay-kernel
ipython kernel install --user --name=predictive-replay-kernel
