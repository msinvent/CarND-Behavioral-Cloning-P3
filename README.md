# **Behavioral Cloning** 

## Project Description

### This project involves training of a deep neural network using the simulated data and then using the model to drive the car autonomously in the same simulator.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data_git.png "Data Generation"
[image2]: ./examples/DNN_Description.png "Model Description"


[//]: # (File References)

[model.py]: ./model.py "model.py"
[drive.py]: ./drive.py "drive.py"
[model.h5]: ./model.h5 "model.h5"
[writeup_report.md]: ./writeup_report.md "project report"
[Autonomous Driving Video]: ./video.mp4 "Autonomous Driving Video"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model ([model.py])
* drive.py for driving the car in autonomous mode ([drive.py])
* model.h5 containing a trained convolution neural network ([model.h5])
* writeup_report.md or writeup_report.pdf summarizing the results ([writeup_report.md])
#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed


I tried to build my model on top of NVIDIA end to end DNN. With dropout layers added in attempt to reduce overfitting and increase robustness of the model.

#### 2. Attempts to reduce overfitting in the model

DNN Model is using dropout to avoid overfitting and further watch was kept on the training and validation accuracy to stop overtraining of the model.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 195).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the data provided by udacity and added some bridge and recovery data points.

For details about how I created the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find a minimal possible model with architecture **similar** to ConvNet and the network used by NVIDIA is of very similar form. My first step was to use a convolution neural network model similar to the **NVIDIA end to end DNN** I thought this model might be appropriate because NVIDIA have already proved its efficacy.

**I started with my own dataset with the recommendation to include smooth driving along with recovery driving by didn't get expected results, so I moved to the standard dataset to make sure that I do not havev a lot of variable to work with (model and data both works as a variable for getting final results).**

In order to gauge how well the model was working, I split the dataset(80 for training and 20% for validation) provided by udacity which is in the form of image and corresponding steering input. I found that my model was showing fast decrement in training error but my validation error with the same rate, showing that model was failing to generalize. Thus I decided to add some dropout layers to avoid overfitting and increase robustness both.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track :
	1 - the bridge
	2 - lane with red and white lane boundaries
	3 - lane with direct transition from road to sand without any strong markings
	4 - Recovery was very inefficient( once the vehicle was going off track, it was not recovering itself to the center of the lane as I would have liked)
	
	I carefully included some more training data from the abover scenarios and trained my model again, with a few hours of experimenting with the model(Yes, I played around with the model architecture, batch size and image cropping area to find the best combination), in the attempt to verify the performance I used the trained model to run the simulater in autonomous mode (which might have lead to some programmer introduced bias, as I should have avoided using the final test results, this may lead to model not generalizing well for extended set of scenarios and I found that my model is not able to pass the track-2 provided with the simulator. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture


The final model architecture (model.py Section 5) consisted of a convolution neural network with the following layers and layer sizes ...

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I started with udacity provided dataset and looked at the failure cases manually to include further data to be included in the dataset. 

**TIPS for including more data : create a git repo and include data in steps rather than attempting to include all the data together, as a lousy driver(simulator driver, not driver driver, infact I am a really safe driver on the road)
That helped me as if I was driving wrong and I didn't wanted to include that particular dataset then I was simply doing *git reset --hard* on the repo
**

#### 4. Generator Class

To feed the network with a dataset larger than my laptop RAM I used generator class concept and inherited from the keras Utils.Sequence class to provide multiprocessing capability for generators to load data parallely. 

**While using jupyter notebook instead of getting a speed increase I was getting speed decrease when using more than 1 processors, I was not able to explore it to the depth due to time limitations but will like to know why and will probably explore it in near future ( meanwhile if you are reading it and know the answer and will like to share the answer then please email me msinvent@gmail.com and I will be very thankful to you.**

The following section attempts to show major sections of the code.

```
class DataGenerator(keras.utils.Sequence):
 ...

training_generator = DataGenerator(X_train, y_train, objectName = 'trainingGenerator', **params)
validation_generator = DataGenerator(X_validation, y_validation, objectName = 'validationGenerator', **params)

model = Sequential()
model.add(...)
...

model.compile(loss='mse',optimizer='adam')
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    shuffle = True,
                    epochs = 8,
                    workers=4,
                    verbose=2)
```

#### 5. Final Results
Well, the vehicle was driving quite well and not breaching the lane boundary, probably the speed is throtelled by the drive.py to 9.0 kph thus the vehicle drops down to 9.0 m/s while driving autonomously.

Sit back and enjoy the final autonomous driving video [Autonomous Driving Video]

#### 6. Future Work
The project is minimal attempt to qualify the requirements set by UDACITY Self-Driving Car Engineer Nano Degree and in no way close to how good it can be in the state of art of autonomous driving sense.

Some modification that I will be taking on in near future (probably in the written order):
1 - Try to use all of my own training data and pass track 1
2 - generalize the model further to pass the track 2.
3 - Remove the speed throttle and then try to pass track 1
4 - Remove the speed throttle and then try to pass track 2


__Enjoy and Drive Safe__
