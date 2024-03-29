library(keras)

# Parameters --------------------------------------------------------------

batch_size <- 32
epochs <- 10
data_augmentation <- TRUE


# Data Preparation --------------------------------------------------------

# See ?dataset_cifar10 for more info
cifar10 <- dataset_cifar10()

# Feature scale RGB values in test and train inputs  
x_train <- cifar10$train$x/255
x_test <- cifar10$test$x/255
y_train <- to_categorical(cifar10$train$y, num_classes = 10)
y_test <- to_categorical(cifar10$test$y, num_classes = 10)

# Defining Model ----------------------------------------------------------

# Initialize sequential model
model <- keras_model_sequential()

model %>%
    
    # Start with hidden 2D convolutional layer being fed 32x32 pixel images
    layer_conv_2d(
        filter = 32, kernel_size = c(3,3), padding = "same", 
        input_shape = c(32, 32, 3)
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
    
    # Outputs from dense layer are projected onto 10 unit output layer
    layer_dense(10) %>%
    layer_activation("softmax")

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = opt,
    metrics = "accuracy"
)


system.time({
  

# Training ----------------------------------------------------------------
data_augmentation = TRUE
if(!data_augmentation){
    print("test")
    model %>% fit(
        x_train, y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = list(x_test, y_test),
        shuffle = TRUE
    )
    
} else {
    print("test2")
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
    
}


})
 
evaluate_generator(model, generator = flow_images_from_data(x_test, y_test, datagen, batch_size = batch_size), steps = batch_size)


g = model %>% predict(x_test[1:220,,,])

get_top <- function(x, n){
    return(which(x == sort(x,T)[n])[1])
}
bb = apply(y_test[1:220,], 1, FUN = function(x)which.max(x))
cc1 = apply(g, 1, FUN = function(x)which.max(x))
cc2 = apply(g, 1, FUN = function(x)get_top(x,2))
cc3 = apply(g, 1, FUN = function(x)get_top(x,3))
cc4 = apply(g, 1, FUN = function(x)get_top(x,4))
cc5 = apply(g, 1, FUN = function(x)get_top(x,5))

mean(bb == cc1)
mean(bb == cc1 | 
         bb == cc2 )
mean(bb == cc1 | 
         bb == cc2 |
         bb == cc3 |
         bb == cc4 |
         bb == cc5
)
     
x = c(1,2,3
      )
which(x==sort(x, T)[2])

# predicted_class <- model %>% predict_generator(generator = flow_images_from_data(x_test, y_test, datagen, batch_size = batch_size), steps = batch_size)
# 
# yp <- apply(predicted_class, 1, FUN = function(x) which.max(x))
# yt2 <- apply(y_test, 1, FUN = function(x) which.max(x))
# yp2 <- apply(predicted_class, 1, FUN = function(x) which(x==sort(x, T)[2]))
# yp3 <- apply(predicted_class, 1, FUN = function(x) which(x==sort(x, T)[3]))
# yp4 <- apply(predicted_class, 1, FUN = function(x) which(x==sort(x, T)[4]))
# yp5 <- apply(predicted_class, 1, FUN = function(x) which(x==sort(x, T)[5]))
# yt <- cifar10$test$y[,1]
# 
# #top 1 accuracy
# mean(yp == yt2)
# mean(yp == yt | yp2 == yt)
# #top 5 accuracy
# mean(yp == yt | yp2 == yt | yp3 == yt | yp4 == yt | yp5 == yt) 
# 
# sort(c(1,2,3,4,5),T)[2]
