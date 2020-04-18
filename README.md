# self_driving_car_TD3

#### Right now, I have proper understanding of both the Deep Q-learning implementation and the TD3 codes.Next step is to integrate both the things together. So, right now I am in the process of integrating both the things. The running code is not ready yet. 

### Things which are done
* the function to get image from the environment based on the current location to be fed to the network. Refer to the ipynb file for implementation details of merging car image on top of sand image.
* the network modules are modifed according to TD3, i.e. convolutional Actor and Critic networks are defined.


### TODO
* Finishing the update function of network file, which will update the replay buffer as well as execute the training of batch after every second.
* Having a different python file named env.py which will take care of all environment related functionalities.
