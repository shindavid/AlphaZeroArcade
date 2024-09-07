# Steps to setup Docker
1. Enable Virtualization on your computer (from your BIOS settings), the instructions depend on your motherboard.
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
3. Install Docker using `apt` ([instructions](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)).
4. Try ```sudo docker run --gpus all hello-world``` to make sure everything works.

# Steps to use this repository in a Docker container
From the main repository folder:
1. Run `sudo ./docker/build.sh` to create the docker image.
2. Run `sudo ./docker/shell` to get a shell to a docker container.
3. Follow the setup instructions in this [README file](https://github.com/shindavid/AlphaZeroArcade/blob/main/README.md).