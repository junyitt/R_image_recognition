# install.packages("KODAMA")
library(KODAMA)
library(Rtsne)
library(dplyr)
library(ggplot2)
library(keras)

getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
}

get_label_from_generator <- function(generator){
    class_indices <- generator$class_indices
    label <- sapply(generator$labels, FUN = function(j){
        names(class_indices[j+1])[1]
    })
    return(label)
}

# Parameters --------------------------------------------------------------
pretrained_model_id <- "LFW4"
path <- "./"
data_path <- file.path(path, "FILTER_Combined")
test_path <- file.path(path, "TEST_FILTER_Combined")

unseen_train_data_path <- file.path(path, "FILTER_UNSEEN_LFW")
unseen_test_data_path <- file.path(path, "TEST_FILTER_UNSEEN_LFW")

model_path <- file.path(path, "Models")
checkpoint_dir <- file.path(path, "Checkpoints")
batch_size <- 32
epochs <- 25

# Load pretrained model ---------------------------------------------------
pretrained_model <- load_model_hdf5(file.path(model_path, paste0("model_", pretrained_model_id, ".h5")))
img_size <- pretrained_model$input_shape[[2]]


# Data Preparation --------------------------------------------------------
train_data_generator <- flow_images_from_directory(data_path, 
                                                   target_size = c(img_size, img_size), 
                                                   shuffle = F
)

# Get the output of the second last layer of the pretrained model ---------
layer_index <- 157
inter_layer <- get_layer(pretrained_model, index = layer_index)
intermediate_layer_model <- keras_model(inputs = pretrained_model$input, outputs = inter_layer$output)

# Get the output of the second last layer on the train data
train_features <- intermediate_layer_model %>% predict_generator(train_data_generator, steps = as.integer(train_data_generator$samples/batch_size))    

# Get the target label of train data
train_label <- get_label_from_generator(train_data_generator)

print(str(train_features))
print(str(train_label))

# Save the features and label for future deployment use
save(train_features, file = "train_features.rda")
save(train_label, file = "train_label.rda")



# Load Unseen Data -------------------------------------------------------------------
# Split the Unseen Data into Training and Testing (Training Data is used to form the feature clusters as a basis for person-reidentification on Testing Data)
unseen_train_data_generator <- flow_images_from_directory(unseen_train_data_path, 
                                                   target_size = c(img_size, img_size), 
                                                   shuffle = F
)
unseen_test_data_generator <- flow_images_from_directory(unseen_test_data_path, 
                                                   target_size = c(img_size, img_size), 
                                                   shuffle = F
)

# Extract the features (128 length vector) on the Unseen Train and Test Data using the pre-trained model
unseen_train_features <- intermediate_layer_model %>% predict_generator(unseen_train_data_generator, steps = as.integer(unseen_train_data_generator$samples/batch_size))
unseen_test_features <- intermediate_layer_model %>% predict_generator(unseen_test_data_generator, steps = as.integer(unseen_test_data_generator$samples/batch_size))
str(unseen_train_features)
str(unseen_test_features)

# Get the target labels 
unseen_train_label <- get_label_from_generator(unseen_train_data_generator)
unseen_test_label <- get_label_from_generator(unseen_test_data_generator)

# Load the train_features and train_label 
load(file = "train_features.rda")
load(file = "train_label.rda")

# Combine both train_features and unseen_train_features; train_label and unseen_train_label
all_train_features <- rbind(train_features, unseen_train_features)
all_train_label <- c(train_label, unseen_train_label)
str(all_train_features)
str(all_train_label)

# Perform K-Nearest Neighbour using the all_train_features and all_train_label to predict the unseen_test_features
ypred <- KODAMA::knn.kodama(Xtrain = all_train_features, Ytrain = as.factor(all_train_label), Xtest = unseen_test_features, k = 7)$Ypred
y_pred <- apply(ypred, 1, getmode) # Majority vote on the label will be the final predicted label

# Accuracy on Unseen Test Data
accuracy_unseen_test <- mean(y_pred == unseen_test_label)
print(paste0("Accuracy (Unseen Test):", accuracy_unseen_test))


# Perform t-SNE to visualize the features extracted from the pre-trained model
all_features <- rbind(all_train_features, unseen_test_features)
all_labels <- c(all_train_label, unseen_test_label)
rstne_result <- Rtsne(all_features, check_duplicates = F)
str(rstne_result$Y) # The output of t-SNE is a 2 dimensional features vector

tsne_features_df <- data.frame(Feature_1 = rstne_result$Y[,1], Feature_2 = rstne_result$Y[,2], Identity = all_labels) 

# Visualize the Features for Train Data
train_label_sample <- unique(all_train_label)[c(1,10,100,110,200,300,400)]
train_data_tsne_plot <- tsne_features_df %>% 
                            filter(Identity %in% train_label_sample) %>%
                            ggplot(aes(x = Feature_1, y = Feature_2, col = Identity)) + 
                            geom_point(shape=2) +
                            ggtitle("High-Dimensional Features Visualization using t-SNE on Training Data")
train_data_tsne_plot # we can observe that the clusters formed are highly distinguishable by the target label

# Visualize the Features for Unseen Test Data
unseen_label_sample <- unique(unseen_test_label)[1:7]
unseen_data_tsne_plot <- tsne_features_df %>% 
    filter(Identity %in% unseen_label_sample) %>%
    ggplot(aes(x = Feature_1, y = Feature_2, col = Identity)) + 
    geom_point(shape=2) +
    ggtitle("High-Dimensional Features Visualization using t-SNE on Unseen Data")
unseen_data_tsne_plot # we can observe that some of the clusters formed can be distinguised by the target label

