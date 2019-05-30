library(keras)

get_class_weights <- function(data_generator){
    counter <- table(data_generator$classes)  
    class_weights <- list()
    IX <- names(counter)
    VA <- counter
    
    for(i in 1:length(IX)){
        class_weights[[IX[i]]] <- VA[[i]]    
    }
    return(class_weights)
}

# Parameters --------------------------------------------------------------
model_id <- "Age"
path <- getwd()
data_path <- file.path(path, "Data/_TRAIN_AgeGroup_UTKFace")
test_path <- file.path(path, "Data/_TEST_AgeGroup_UTKFace")
model_path <- file.path(path, "Models")
checkpoint_dir <- file.path(path, "Checkpoints")
input_shape_len <- 128
batch_size <- 32
epochs <- 50
learning_rate <- 0.00003

dir.create(model_path, showWarnings = FALSE)
dir.create(checkpoint_dir, showWarnings = FALSE)

# Data Preparation --------------------------------------------------------
image_data_generator_1 <- image_data_generator(
    rotation_range = 30, #20
    width_shift_range = 0.3, #0.2
    height_shift_range = 0.3, #0.2
    brightness_range = c(0.1, 1.0),
    channel_shift_range = 0.01,
    zoom_range = 0.3,
    horizontal_flip = TRUE, 
    validation_split = 0.2
)

train_data_generator <- flow_images_from_directory(data_path, 
                                             generator = image_data_generator_1,
                                             target_size = c(input_shape_len, input_shape_len), subset = "training"
)

val_data_generator <- flow_images_from_directory(data_path, 
                                                   generator = image_data_generator_1,
                                                   target_size = c(input_shape_len, input_shape_len), subset = "validation"
)

cw <- get_class_weights(train_data_generator)


input_img <- layer_input(shape = c(input_shape_len, input_shape_len, 3))
num_classes <- train_data_generator$num_classes

# Load and Initialize Model --------------------------------------------------------

model <- keras_model_sequential()

model %>%
    # Start with hidden 2D convolutional layer being fed 32x32 pixel images
    layer_conv_2d(
        filter = 32, kernel_size = c(3,3), padding = "same", 
        input_shape = c(input_shape_len, input_shape_len, 3)
    ) %>%
    layer_activation("relu") %>%
    
    # Second hidden layer
    layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
    layer_activation("relu") %>%
    
    # Use max pooling
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(0.25) %>%
    
    # 2 additional hidden 2D convolutional layers
    layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
    layer_activation("relu") %>%
    layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
    layer_activation("relu") %>%
    
    # Use max pooling once more
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(0.25) %>%
    
    # Flatten max filtered output into feature vector 
    # and feed into dense layer
    layer_flatten() %>%
    layer_dense(512) %>%
    layer_activation("relu") %>%
    layer_dropout(0.5) %>%
    layer_dense(units = num_classes, activation = 'softmax')
# model <- keras_model(inputs = base_model$input, outputs = predictions)

opt <- optimizer_rmsprop(lr = learning_rate, decay = 1e-6)

model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = opt,
    metrics = "accuracy"
)

# Initialize Checkpoints --------------------------------------------------------
checkpoint_filepath <- file.path(checkpoint_dir, paste0("weights_", model_id, "_{epoch:02d}-{val_loss:.2f}.hdf5"))

cp_callback <- callback_model_checkpoint(
    filepath = checkpoint_filepath,
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 1
)

# Training ----------------------------------------------------------------
system.time({
    model %>% fit_generator(
        train_data_generator,
        validation_data = val_data_generator,
        steps_per_epoch = as.integer(train_data_generator$samples/batch_size), 
        validation_steps  = as.integer(val_data_generator$samples/batch_size), 
        callbacks = list(cp_callback), 
        class_weight = cw,
        epochs = epochs
    )
})

checkpoints_w <- list.files(checkpoint_dir, pattern = paste0("weights_", model_id))
last_checkpoint <- checkpoints_w[length(checkpoints_w)]

# Load best model
model %>% load_model_weights_hdf5(
    file.path(checkpoint_dir, last_checkpoint)
)

class_indices <- train_data_generator$class_indices

# Save Best Model and Metadata ----------------------------------------------------------------
save(class_indices, file = file.path(model_path, paste0("/class_indices_", model_id, ".rdata"))) #save class_indices
model %>% save_model_hdf5(file.path(model_path, paste0("model_", model_id, ".h5"))) #save model

