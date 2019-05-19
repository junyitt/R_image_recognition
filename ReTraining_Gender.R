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
pretrained_model_id <- "LFW2"
model_id <- "UTKFace_Gender_3"
path <- "C:/Users/jy/Desktop/R_IR_7004/"
data_path <- file.path(path, "FILTER_Gender_UTKFace")
test_path <- file.path(path, "TEST_FILTER_Gender_UTKFace")
model_path <- file.path(path, "Models")
checkpoint_dir <- file.path(path, "Checkpoints")
# img_size <- 128
batch_size <- 32
epochs <- 25
learning_rate <- 0.0002

# Load pretrained model
pretrained_model <- load_model_hdf5(file.path(model_path, paste0("model_", pretrained_model_id, ".h5")))
img_size <- pretrained_model$input_shape[[2]]


# Data Preparation --------------------------------------------------------
image_data_generator_1 <- image_data_generator(
    rotation_range = 30, #20
    width_shift_range = 0.3, #0.2
    height_shift_range = 0.3, #0.2
    horizontal_flip = TRUE, 
    validation_split = 0.2
)

train_data_generator <- flow_images_from_directory(data_path, 
                                             generator = image_data_generator_1,
                                             target_size = c(img_size, img_size), subset = "training"
)

val_data_generator <- flow_images_from_directory(data_path, 
                                                   generator = image_data_generator_1,
                                                   target_size = c(img_size, img_size), subset = "validation"
)

test_data_generator <- flow_images_from_directory(test_path, 
                                                 generator = image_data_generator_1,
                                                 target_size = c(img_size, img_size)
)


# Get Class Weights
cw <- get_class_weights(train_data_generator)

input_img <- layer_input(shape = c(img_size, img_size, 3))
num_classes <- train_data_generator$num_classes

# Get the output of the second last layer of the pretrained model
layer_index <- 158
inter_layer <- get_layer(pretrained_model, index = layer_index)
# pretrained_cut_model <- keras_model(inputs = pretrained_model$input, outputs = inter_layer$output)
# Add the output layer (Now only 2 classes - male/female)
predictions <- inter_layer$output %>% 
                layer_dense(units = num_classes, activation = 'softmax')
model <- keras_model(inputs = pretrained_model$input, outputs = predictions)

layers <- model$layers
for (i in 1:length(layers)){
    cat(i, layers[[i]]$name, "\n")
}

opt <- optimizer_rmsprop(lr = learning_rate, decay = 1e-6)

model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = opt,
    metrics = "accuracy"
)

dir.create(checkpoint_dir, showWarnings = FALSE)
checkpoint_filepath <- file.path(checkpoint_dir, paste0("weights_", model_id, "_{epoch:02d}-{val_loss:.2f}.hdf5"))

cp_callback <- callback_model_checkpoint(
    filepath = checkpoint_filepath,
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 1
)

system.time({
    # Training ----------------------------------------------------------------
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
save(class_indices, file = file.path(model_path, paste0("/class_indices_", model_id, ".rdata"))) #save class_indices
model %>% save_model_hdf5(file.path(model_path, paste0("model_", model_id, ".h5"))) #save model



test_data_generator <- flow_images_from_directory(test_path, 
                                                 generator = image_data_generator_1,
                                                 target_size = c(img_size, img_size)
)

model %>% evaluate_generator(test_data_generator, steps = as.integer(test_data_generator$samples/batch_size))

