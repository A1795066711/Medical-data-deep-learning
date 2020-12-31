# Medical-data-deep-learning

## Documents

### CNN.py
Orginal CNN model

### D2T4 CNN Tensroflow_1300hidden.py, D2T4 CNN Tensroflow_1800hidden.py
Modified CNN model with hidden layer size at 1300 and 1800 respectively

### WISDM_ar_v1.1_raw.zip
File that contains the data set as a form of TXT

### Distribution
Data set distribution on each movement category 

## Project Intro
With a data set from a contest, the project is aiming at improving the accuracy of the CNN models applied to medical data.

Each entry of the medical data set is organized in following format:

                           { 33,           Jogging,            49105962326000,  -0.6946377,                  12.680544,                   0.50395286 }
                           { Patient ID,   Movement( label ),  Timestamp,       Partial accelerate(X axis),  Partial accelerate(Y axis),  Partial accelerate(Z axis) } 
                           
### The general CNN model
The structure and some attributes are represented in CNN.pdf 
