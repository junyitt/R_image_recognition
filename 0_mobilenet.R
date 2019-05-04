library(keras)

# Parameters --------------------------------------------------------------

batch_size <- 32
epochs <- 30
data_augmentation <- TRUE


# Data Preparation --------------------------------------------------------

# See ?dataset_cifar10 for more info
cifar10 <- dataset_cifar10()

# Feature scale RGB values in test and train inputs  
x_train <- cifar10$train$x/255
x_test <- cifar10$test$x/255
y_train <- to_categorical(cifar10$train$y, num_classes = 10)
y_test <- to_categorical(cifar10$test$y, num_classes = 10)


input_img <- layer_input(shape = c(32, 32, 3))


    # Load pre-trained mobilenetv2
    # base_model <- application_mobilenet_v2(include_top = F, weights = "imagenet", input_tensor = input_img)
    # 
    # # add our custom layers
    # predictions <- base_model$output %>% 
    #     keras::layer_global_average_pooling_2d() %>% 
    #     # layer_dense(units = 1024, activation = 'relu') %>%
    #     layer_dense(units = 10, activation = 'softmax')
    # 
    # # this is the model we will train
    # model <- keras_model(inputs = base_model$input, outputs = predictions)
    # 
    # base_model <- application_mobilenet_v2(include_top = F, input_tensor = input_img)
    base_model <- application_mobilenet_v2(include_top = F, weights = "imagenet", input_tensor = input_img)
    predictions <- base_model$output %>% 
            keras::layer_global_average_pooling_2d() %>% 
            layer_dense(units = 10, activation = 'softmax')
    model <- keras_model(inputs = base_model$input, outputs = predictions)
    
    # model2 <- application_mobilenet_v2(include_top = T, weights = "imagenet", input_tensor = input_img)
    summary(model)
    # Show layers
    layers <- model$layers
    for (i in 1:length(layers)){
        cat(i, layers[[i]]$name, "\n")
    }
    
    # Freeze weights
    freeze_weights(model, from = 1, to = 64)
    
    summary(model)

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = opt,
    metrics = "accuracy"
)


system.time({
    # Training ----------------------------------------------------------------
    datagen <- image_data_generator(
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = TRUE
    )
    
    datagen %>% fit_image_data_generator(x_train)
    
    model %>% fit_generator(
        flow_images_from_data(x_train, y_train, datagen, batch_size = batch_size),
        steps_per_epoch = as.integer(50000/batch_size), 
        epochs = epochs,
        validation_data = flow_images_from_data(x_test, y_test, datagen, batch_size = batch_size)
    )
})

# 
# model %>% fit_generator(
#     flow_images_from_data(x_train, y_train, datagen, batch_size = batch_size),
#     steps_per_epoch = as.integer(50000/batch_size), 
#     epochs = 20,
#     validation_data = flow_images_from_data(x_test, y_test, datagen, batch_size = batch_size)
# )
