mkdir -p ./.carla
wget -N https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.9.4.tar.gz
tar xzf ./CARLA_0.9.9.4.tar.gz -C ./.carla
wget -N http://54.201.45.51:5000/mtp/gnn.tar.gz
tar xzf ./gnn.tar.gz -C .
wget -N http://54.201.45.51:5000/mtp/config.tar.gz
tar xzf ./config.tar.gz -C .