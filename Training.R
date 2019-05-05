library(keras)

# Parameters --------------------------------------------------------------

batch_size <- 32
epochs <- 5

# Data Preparation --------------------------------------------------------
image_data_generator_1 <- image_data_generator(
    rotation_range = 45, #20
    width_shift_range = 0.4, #0.2
    height_shift_range = 0.4, #0.2
    horizontal_flip = TRUE, 
    validation_split = 0.2
)

train_data_generator <- flow_images_from_directory("C:/Users/jy/Desktop/R_IR_7004/faces-data-new/main_data", 
                                             generator = image_data_generator_1,
                                             target_size = c(180, 180), subset = "training"
)

val_data_generator <- flow_images_from_directory("C:/Users/jy/Desktop/R_IR_7004/faces-data-new/main_data", 
                                                   generator = image_data_generator_1,
                                                   target_size = c(180, 180), subset = "validation"
)




input_img <- layer_input(shape = c(180, 180, 3))
num_classes <- train_data_generator$num_classes

base_model <- application_mobilenet_v2(include_top = F, input_tensor = input_img)
predictions <- base_model$output %>% 
    keras::layer_global_average_pooling_2d() %>% 
    layer_dense(units = num_classes, activation = 'softmax')
model <- keras_model(inputs = base_model$input, outputs = predictions)

layers <- model$layers
for (i in 1:length(layers)){
    cat(i, layers[[i]]$name, "\n")
}

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = opt,
    metrics = "accuracy"
)


system.time({
    # Training ----------------------------------------------------------------
    model %>% fit_generator(
        train_data_generator,
        validation_data = val_data_generator,
        steps_per_epoch = as.integer(train_data_generator$samples/batch_size), 
        validation_steps  = as.integer(val_data_generator$samples/batch_size), 
        epochs = epochs
    )
})


model_id <- "1"
model_path <- "C:/Users/jy/Desktop/R_IR_7004/Models/"
class_indices <- data_generator$class_indices
save(class_indices, file = paste0(model_path, "class_indices_", model_id, ".rdata")) #save class_indices
model %>% save_model_hdf5(paste0(model_path, "model_", model_id, ".h5")) #save model
