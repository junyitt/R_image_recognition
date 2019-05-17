# Libraries ---------------------------------------------------------------
library(keras)
library(densenet)

# Parameters --------------------------------------------------------------

batch_size <- 64
epochs <- 300

# Data Preparation --------------------------------------------------------

# see ?dataset_cifar10 for more info
cifar10 <- dataset_cifar10()

# Normalisation
for(i in 1:3){
    mea <- mean(cifar10$train$x[,,,i])
    sds <- sd(cifar10$train$x[,,,i])
    
    cifar10$train$x[,,,i] <- (cifar10$train$x[,,,i] - mea) / sds
    cifar10$test$x[,,,i] <- (cifar10$test$x[,,,i] - mea) / sds
}
x_train <- cifar10$train$x
x_test <- cifar10$test$x

y_train <- to_categorical(cifar10$train$y, num_classes = 10)
y_test <- to_categorical(cifar10$test$y, num_classes = 10)

# Model Definition -------------------------------------------------------

input_img <- layer_input(shape = c(32, 32, 3))
# model <- application_densenet(include_top = TRUE, input_tensor = input_img, dropout_rate = 0.2)
# base_model <- application_mobilenet_v2(include_top = F, input_tensor = input_img, classes = 10)
base_model <- application_mobilenet_v2(include_top = F, weights = "imagenet", input_tensor = input_img)
base_model2 <- application_mobilenet_v2(include_top = T, weights = "imagenet", input_tensor = input_img)


# add our custom layers
predictions <- base_model$output %>% 
    keras::layer_global_average_pooling_2d() %>% 
    # layer_dense(units = 1024, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')

# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)


# Show layers
layers <- model$layers
for (i in 1:length(layers)){
    cat(i, layers[[i]]$name, "\n")
}

freeze_weights(model, from = 1, to = 155)

summary(model)


opt <- optimizer_sgd(lr = 0.1, momentum = 0.9, nesterov = F)
# opt <- optimizer_adam(lr = 0.0001)
# opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
    optimizer = opt,
    loss = "categorical_crossentropy",
    metrics = "accuracy"
)

# Model fitting -----------------------------------------------------------

# callbacks for weights and learning rate
lr_schedule <- function(epoch, lr) {

    if(epoch <= 7) {
        0.1
    } else if(epoch > 7 && epoch <= 30){
        0.01
    } else {
        0.001
    }

}


lr_reducer <- callback_learning_rate_scheduler(lr_schedule)

system.time({
    history <- model %>% fit(
        x_train, y_train, 
            
        batch_size = batch_size, 
        epochs = epochs, 
        validation_data = list(x_test, y_test), 
        callbacks = list(
            lr_reducer
        )
    )
})

evaluate(model, x_test, y_test)


