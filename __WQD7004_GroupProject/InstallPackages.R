# Install Anaconda
# Recommended to install: Anaconda3-2018.12-Windows-x86_64.exe
# https://repo.anaconda.com/archive/

# Install OpenCV
# On Anaconda
# !pip install opencv-python 
# !pip install numpy

# Install R Packages
install.packages("reticulate")
install.packages("stringr")
install.packages("devtools")
install.packages("caret")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("Rtsne")
install.packages("KODAMA")
install.packages("shiny")
install.packages("shinycssloaders")

# Install Rtools 3.5 (Rtools35.exe)
# http://cran.r-project.org/bin/windows/Rtools/

# Install Keras and Tensorflow GPU
devtools::install_github("rstudio/keras")
devtools::install_github("rstudio/tensorflow")
library(keras) 
install_keras(tensorflow = "gpu") #for gpu version
install_tensorflow(version = "gpu")

