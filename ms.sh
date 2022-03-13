git config --global user.name 'namespace-Pt'
git config --global user.email 'namespace.pt@gmail.com'
conda init
echo 'alias nn="conda activate /data/v-pezhang/Envs/nn"' >> ~/.bashrc
echo 'alias pt="screen -r -d pt"' >> ~/.bashrc
echo 'alias ns="nvidia-smi"' >> ~/.bashrc

source ~/.profile
source ~/.bashrc
sudo apt-get -y install screen

# sudo apt-get -y install rsync
cd /data/v-pezhang/Code/GateFormer/src