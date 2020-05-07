# Self Driving Car using Reinforcement Learning(Twin Delayed DDPG)

### Problem Statement:
Train a car to move from one source point to destination point following the give proper road pathway covering the minimum distance in a map.
### Completed important steps
* Written the full end-to-end training process.
* Passed the cropped image at the exact position, along with a arrow like structure to show the direction of the car, so that the network can understand the direction of car better. Example image is shown below
<img src="https://i.imgur.com/H3ylehR.png">

* For the progress of training, was able to successfully train the network to make the car reach from one source to destination ignoring the sand obstacle.
* Tried with multiple variations of the reward order to make the car keep in the sand. Which is not achieved yet. I am missing something or doing some mistake in terms of the direction of the car because of different co-ordinate of origin for kivy and pillow system.
* The only input I am using is image. I am not sending any orientation imformation. The reason being I want the network to learn everything from the image information alone.


### Summary of all the experiments and output results till now:
* One issue I have realized with pillow to numpy conversion as well as the origin of the co-ordinates of kivy and pillow. Basically because of this issue, I followed a differnt approach from the approach of handling images in session 7. When we use the horizontal image itself I found it is easier to locate the sand value to panalize the reward system just using the complement of y value by replacing y values with (largeur - y) value.

### Explanation of Each of the individual files
#### tesla.kv
This is the kv file, where the visualisation parameters of the environment is defined.

#### main.py
##### The main file where the "Game" widget is called for running the environment. The main componets of the file are explained below.

**get_input_image** : The function to crop the image patch from the map containing the car, using the position of the car information. This image will be our state variable for the TD3 algorithm.

**Car** : This block is to define the car and updating the pos and velocity parameters of the car to make it moving.

**Game** : This is the back-bone of the app. The update fuction of this class is getting called at every second and it gets the current state and passes it to the brain block of network.py file, which is used for training the network and at the same time it updates the environment using action returned by the network. Based on the action the reward is also calcuated and accumulated for training and tracking purpose. The action only makes the car moving.
We are having on "done_bool" variable, which is keeping track if one episode is done based on two parameters if the car reached the final destination by calculating distance or if the total reward for that particualar episode is not improving and it went beyoing certain total reward point. If either of the conditions are satisfied then the environmnent is reset and a new episode is started with a random starting point and this is how this process continues. Whenever one episodes gets over, the done_bool variable is used to trigger the training. The done_bool variable is passed to the network.py file.

#### network.py
##### The file which has the architecture for the basic building networks of The TD3 model named Anchor and Critic. The training strategy is also defined in this file.

**Actor, Target** : Definition of the networks. In both the cases the image embedding is calculated using covolution blocks and the concatenated with extra information followed by linear layers to get the final output. 

**ReplayBuffer** : The class to construct the replay buffer for saving the transition which will be randomly sampled during the time of training.

**TD3**:
*update* = This is the function which is called from the main.py for every update. For first few iterations, the action is choosen randomly to populate the replay_buffer and after that the action is calucated based on the state variable where we are sending the cropped image along with the orientation of the car. In this step only we are updating the replay_buffer.
*learn* = When one episode is done, this learn method is called to start the training process, where the batches are sampled from the replay_buffer. We are following the TD3 alogorithm for learning process.

### Important Parameters:
* state_dim = input_image(1x40x40) + orientation(1)
* action_dim = 1
* max_action = 10 ## Not confident about this parameter. Have to do some more experiments to come up with the correct value

### Progress Till Now:
* Implemented the whole flow end to end where the TD3 learning strategy is implemented along with the necessary blocks.
* The car can be visualised in the map, when the training is going on.

### Current Issues:
* The action which is returned by the Actor model is not changing, from the previous value because of which when the random actions are done taking the car starts rotating in around iteself.
* Not sure about the correct value of the total accumulated reward, when that minimum value is crossed then the episode has to be stopped.

### To do:
* Figuring out the problem of having the same value of action. Have to find out the temperature value along with the correct value of max_action to get to the correct action value. After that the noise should be added to the action.
* Playing with the rewards value to get to the optimized point of training.
* Writing the functions to save and load the model for continuing the training and function to evaluate the training process.
* Writing inference moduel where the final output can be visualised given any random start and end point using pretrained model.
