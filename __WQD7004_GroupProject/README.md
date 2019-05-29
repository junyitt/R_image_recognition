# WQD7004: PROGRAMMING FOR DATA SCIENCE 
# Group Project

## Title
### TRAINING A CONVOLUTIONAL NEURAL NETWORK TO DO FACIAL RECOGNITION USING BIG DATA 

## Group Members: 
1. Cheah Jun Yitt (WQD180107)
2. Tan Yin Yen (WQD180108)
3. Choo Jian Wei (WQD180124)

## README
- Make sure all software and hardware dependencies have been met. (*List of dependencies can be found in the Group Report.)  
- Refer to InstallPackages.R for the list of commands to install the needed packages.  
- Setting up GPU drivers, CUDA and instaling Keras on R can be tricky. Make sure to follow the instructions on <https://keras.rstudio.com/>. Do test out a few Keras examples listed on <https://keras.rstudio.com/articles/examples/index.html>.   
- **Training the CNN models using CPU might not work! Training was done on a GTX1070 GPU.**  

- Download the necessary data, extract and place them in the Data folder.
    - Data can be downloaded here:
        - Labeled Faces in the Wild (LFW): <http://vis-www.cs.umass.edu/lfw/#deepfunnel-anchor>
        - faces_data_new: <https://www.kaggle.com/gasgallo/faces-data-new>
        - UTKFace: <https://susanqq.github.io/UTKFace/>  
*Note: For WQD7004 Group Project submission, training will be done only a small subset on the data selected manually. This is to allow for training on CPU.
    
- Run the "RUN.R" script to pre-process data and train the models from scratch. At the end of the script, a Shiny app will be deployed to allow user to visualize the output of the trained models.

