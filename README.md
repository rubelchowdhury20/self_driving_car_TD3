# Self Driving Car using Reinforcement Learning(Twin Delayed DDPG)

### Problem Statement:
Train a car to move from one source point to destination point following the give proper road pathway covering the minimum distance in a map.

### Explanation of Each of the individual files
#### tesla.kv
This is the kv file, where the visualisation parameters of the environment is defined.

#### main.py
##### The main file where the "Game" widget is called for running the environment. The main componets of the file are explained below.

*get_input_image* : The function to crop the image patch from the map containing the car, using the position of the car information. This image will be our state variable for the TD3 algorithm.

*Car* : This block is to define the car and updating the pos and velocity parameters of the car to make it moving.

*Game* : This is the back-bone of the app. The update fuction of this class is getting called at every second and it gets the current state and passes it to the brain block of network.py file, which is used for training the network and at the same time it updates the environment using action returned by the network. Based on the action the reward is also calcuated and accumulated for training and tracking purpose. The action only makes the car moving.
We are having on "done_bool" variable, which is keeping track if one episode is done based on two parameters if the car reached the final destination by calculating distance or if the total reward for that particualar episode is not improving and it went beyoing certain total reward point. If either of the conditions are satisfied then the environmnent is reset and a new episode is started with a random starting point and this is how this process continues. Whenever one episodes gets over, the done_bool variable is used to trigger the training. The done_bool variable is passed to the network.py file.

#### network.py
##### The file which has the architecture for the basic building networks of The TD3 model named Anchor and Critic. The training strategy is also defined in this file.

*ReplayBuffer* : The class to construct the replay buffer for saving the transition which will be randomly sampled during the time of training.

