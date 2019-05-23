library(keras)


get_label_from_generator <- function(generator){
    class_indices <- generator$class_indices
    label <- sapply(generator$labels, FUN = function(j){
        names(class_indices[j+1])[1]
    })
    return(label)
}

# Parameters --------------------------------------------------------------
pretrained_model_id <- "LFW2"
path <- "C:/Users/jy/Desktop/R_IR_7004/"
data_path <- file.path(path, "FILTER_Combined")
test_path <- file.path(path, "TEST_FILTER_Combined")
model_path <- file.path(path, "Models")
checkpoint_dir <- file.path(path, "Checkpoints")
batch_size <- 32
epochs <- 25

# Load pretrained model
pretrained_model <- load_model_hdf5(file.path(model_path, paste0("model_", pretrained_model_id, ".h5")))
img_size <- pretrained_model$input_shape[[2]]


# Data Preparation --------------------------------------------------------
train_data_generator <- flow_images_from_directory(data_path, 
                                                   target_size = c(img_size, img_size), 
                                                   shuffle = F
)


# Get the output of the second last layer of the pretrained model
layer_index <- 157
inter_layer <- get_layer(pretrained_model, index = layer_index)
intermediate_layer_model <- keras_model(inputs = pretrained_model$input, outputs = inter_layer$output)

system.time({
    train_features <- intermediate_layer_model %>% predict_generator(train_data_generator, steps = 2*as.integer(train_data_generator$samples/batch_size))    
})
train_label <- get_label_from_generator(train_data_generator)
print(dim(train_features))
print(str(train_label))

save(train_features, file = "train_features.rda")
save(train_label, file = "train_label.rda")


unseen_train_data_path <- file.path(path, "FILTER_UNSEEN_LFW")
unseen_test_data_path <- file.path(path, "TEST_FILTER_UNSEEN_LFW")
unseen_train_data_generator <- flow_images_from_directory(unseen_train_data_path, 
                                                   target_size = c(img_size, img_size), 
                                                   shuffle = F
)
unseen_test_data_generator <- flow_images_from_directory(unseen_test_data_path, 
                                                   target_size = c(img_size, img_size), 
                                                   shuffle = F
)

system.time({
    unseen_train_features <- intermediate_layer_model %>% predict_generator(unseen_train_data_generator, steps = 2*as.integer(unseen_train_data_generator$samples/batch_size))
    unseen_test_features <- intermediate_layer_model %>% predict_generator(unseen_test_data_generator, steps = 2*as.integer(unseen_test_data_generator$samples/batch_size))
    
})
str(unseen_train_features)
str(unseen_test_features)

unseen_train_label <- get_label_from_generator(unseen_train_data_generator)
unseen_test_label <- get_label_from_generator(unseen_test_data_generator)

# install.packages("KODAMA")

library(KODAMA)
load(file = "train_features.rda")
load(file = "train_label.rda")
all_train_features <- rbind(train_features, unseen_train_features)
all_train_label <- c(train_label, unseen_train_label)
str(all_train_features)
str(all_train_label)
ypred <- KODAMA::knn.kodama(Xtrain = all_train_features, Ytrain = as.factor(all_train_label), Xtest = unseen_test_features, k = 7)$Ypred

getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
}

y2 <- apply(ypred, 1, getmode)
z <- y2 == unseen_test_label
mean(z)
data.frame(y2, unseen_test_label)
