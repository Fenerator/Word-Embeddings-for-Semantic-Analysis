# Setup: Serbian Semantic Textual Similarity

This document provides you with instructions for how to set up a python virtual environment on a remote server with all the necessary packages for our task.

You will interact with the server via the command line. The most important commands are given, but in some cases you will need to do some research yourself if you're unfamiliar with working via the command line.

## Access to server

**Important:** Make sure you are connected to the UZH VPN before logging in to the server.

Log in to your personal folder:

`ssh [username]@[IP-address]`

You will be asked to enter your password, which you should have received separately by email.

Add the server to the list of trusted hosts when prompted.

You are free to explore the server's folder structure to get an overview.

## Setting up your personal folder

First, create some directories inside your personal folder called "venvs" and "scripts".

## Setting up a virtual environment

Create a new virtual environment using the python `venv` module in the venvs folder called "transformers". Be sure to use `python3`.

**Important**: Activate the newly created virtual environment.

**Upgrade pip**

This is important for some of the packages that are installed later.

`pip install -U pip`

**Install the following packages using pip**

pandas tqdm scikit-learn gensim

**Install PyTorch**

Check out https://pytorch.org/get-started/locally/ for more information about the installation with CUDA (you only need torch, not torchvision or torchaudio).

**Install the Transformers libraries**

transformers simpletransformers

Finally, check that all packages and dependencies were installed using pip. If something went wrong, you can always delete the virtual environment folder and create a fresh one (beware however that the `rm` command removes files and folder irrevocably, so be careful with it).
