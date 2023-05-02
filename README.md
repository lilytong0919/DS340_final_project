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


What each file do (roughly):

theAI.py: the function you should run, it contains the main, the Agent class, all plotting helper, and the training function (and its helper).

theModel.py: this contains the Linear_QNet class (the neural network used by the agent) and the QTrainer class (the function that trains the agent, called by train_short_memory and train_long_memory, where train_long_memory is basically memory replay).

WaterMazeAI.py: this includes the WaterMazeAI class, which setup the game environment that simulate the Morris Water Maze. Reward is setted inside play_step().

figures: the folder I used to save plotted figues in. Saving is disabled as I commented out those part in the submitted version. But this folder also cotains record of my past attempts, with semi-documented network settings (meaning I said in the file names what I did, and inconsistently include a .txt file that recorded the parameters)

models: the folder I used to save trained models, its not worth looking, I stopped saving them. (For some unknown reason I haven't looked into every time I run a saved model it don't work. But also when I still tries to keep the models I trained they don't learn).

interaction_with_chatGPT.pdf: the embarrassingly long conversation with chatGPT about how to write my project. It's so long that if I try to put the text directly into my paper appendex my paper becomes 41 pages long. So it is kept here.
