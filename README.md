# DS340_final_project
The repository for data science course final project, a DQN learner on hand made Morries Water Maze

To run the code just run theAI.py. Hopefully it set-up correctly to run on computers that are not of my own. (To ensure that I disabled any form of saving).


It is likely that some of the libraries I used (proabably numpy and pytorch) fight against each other for some reason that I do not know. But I do know how to stop them from fighting.
So here is the environment setup:
0. Basically follow what this guy did in his youtube video from 1:02 to 3:14, but remember to add numpy after doing whatever he did: https://www.youtube.com/watch?v=5Vy5Dxu7vDs&list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV&index=2
1. With anacondaprompt, start a new environemnt: conda create -n env_name python=3.7
2. Activate the environment: conda activate env_name
3. Use pip (not conda) install to get these packages: pygame, numpy, matplotlib, ipython
4. go to https://pytorch.org/get-started/locally/, install pytorch with the following option: Stable, your own OS, pip, python, cpu. And then copy the command given in "Run this command" section.

It should be working with this setup, but if it doesn't the best I know is to do it over again. (Unless a package is missing then just install it with pip install (not conda!)
