# Title: TRAINING A CONVOLUTIONAL NEURAL NETWORK TO DO FACIAL RECOGNITION USING BIG DATA 
#
# Group Members: 
# 1. Cheah Jun Yitt (WQD180107)
# 2. Tan Yin Yen (WQD180108)
# 3. Choo Jian Wei (WQD180124)
#
##########
# README
##########
# Please read the README.md before running this R script.
# This R Script allows you to run the entire Project. All steps from Data Pre-processing to Deployment were divided into each separate R scripts.
#*Note: Please run the following line by line. Do not run all lines at once. (There is a need to restart the R session on RStudio.)

print("Start: Training a CNN for Image Recognition.")
############################################################
# Pre-process and Filter Data
############################################################
# Training Data for Identity Model 
source("./Processes/PreProcess_Data_LFW.R") 
source("./Processes/PreProcess_Data_faces_data_new.R")
    
# Training Data for Gender Model 
source("./Processes/PreProcess_Data_Gender_UTKFace.R")
    
# Training Data for Age Model 
source("./Processes/PreProcess_Data_Age_UTKFace.R")

# Training Data for Person Re-Identification
source("./Processes/PreProcess_Data_LFW_PersonReID.R")

print("Completed: Data Pre-processing")

############################################################
# Split Data into Training and Testing 
############################################################
# Extract 20% Test Data from Training Data
source("./Processes/SplitTestData_LFW.R")
source("./Processes/SplitTestData_LFW_PersonReID.R")
source("./Processes/SplitTestData_faces_data_new.R")
source("./Processes/SplitTestData_Gender_UTKFace.R")
source("./Processes/SplitTestData_Age_UTKFace.R")

# Combine both LFW and FDN Data into a Combined Data for Identity Recognition Model
source("./Processes/PreProcess_Combine_LFW_FDN.R")

############################################################
# Train the Facial Recognition Models
############################################################
.rs.restartR() # Restart the R Session (R must be restarted to reload the Python Environment for Keras)
library(reticulate)
reticulate::use_condaenv(condaenv = 'r-tensorflow', required = TRUE)

    #########################################
    # *TRAINING MIGHT NOT WORK ON CPU
    #########################################
    source("./Processes/Training_Identity.R")
    source("./Processes/Training_Gender.R")
    source("./Processes/Training_AgeGroup.R")

print("Done: Training a CNN for Image Recognition.")

############################################################
# Evaluate Models
############################################################
#**The model performance might be very low if you are using a small subset of the data.
source("./Processes/Evaluate_Identity.R")
source("./Processes/Evaluate_Gender.R")
source("./Processes/Evaluate_AgeGroup.R")
source("./Processes/Evaluate_PersonReIdentification.R")
# Visualize features cluster via t-sne plots
train_data_tsne_plot 
unseen_data_tsne_plot

print("Done: Model Evaluation.")

############################################################
# Deployment of Models as a Facial Recognition Application
############################################################
source("./Processes/Deployment.R")
