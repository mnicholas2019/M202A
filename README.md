# Activity Classification Using In-Ear Wearables


## Project Overview

The goal of our project is to utilize the eSense in-ear wearable device and other wearable devices, such as a wrist band, to classify and detect a set of user activities, such as smoking or drinking. 

## Approach
Input data will be recorded from the microphone and 6-axis intertial measurement unit (IMU) in the eSense device, and IMU in the wrist band. Traditional signal procesing approaches, including a  complementary/Kalman filter, will be implemented. Filtered data will be used to train a Convolutional Neural Network (CNN) for activity classification. A method of storing user activity data will also be implemented, allowing data analysis such as activity frequency and temporal patterns.  

## Deliverables
* Trained CNN with sufficient accuracy in classifying several similar activities using filtered IMU/audio data 
* Android Application with real-time activity classification

## Timeline
* Week 4
  * Finalized project objective and initial plan of attack
  * Obtain wearable hardware and prior software
* Week 5
  * Study prior research and install software (preexisting wearable data collection, Tensorflow/Keras, Android Studio)
  * Create outline of software architecture and finalize project deliverables/goals
* Week 6
  * Collect sensor data and assign truth data (which activity the user is currently performing)
  * Begin signal processing and training CNN
* Week 7/8
  * Present on preliminary design/results
  * Continue training and improving classification model
  * Begin Android app design and development
* Week 9
  * Finalize classification model
  * Continue working on Android real-time classification app
* Week 10-12
  * Reevaluate project goals vs status of project, determine best use of time to successfully submit a working project
  * Write final report/presentation, record/edit final video
* Week 13
  * Submit 25-minute YouTube video outlining project lifecycle and success
  * Wrap up final submission including: report, slides, code, and return hardware
  
## Division of Work
Since we are a group of two, most subsections of the project will be developed together. However, each member is assigned the following sections to oversee:
* Aidan Cookson
  * Complementary/Kalman Filter 
  * Classification data collection software
  * Android Application BT Communication with sensors
* Matt Nicholas
  * Additional Signal Processing  
  * CNN model structure/implementation
  * Android Application UI/integration with CNN
