# Libraries ---------------------------------------------------------------
library(keras)
library(densenet)


model = application_vgg16(include_top = TRUE, weights = "imagenet",
                  input_tensor = NULL, input_shape = NULL, pooling = NULL,
                  classes = 1000)
# model2 = application_vgg16(include_top = TRUE, weights = "imagenet",
#                           input_tensor = NULL, input_shape = NULL, pooling = NULL,
#                           classes = 1001)
application_vgg16


img_path <- "som.jpg"
img <- image_load(img_path, target_size = c(224,224))
x <- image_to_array(img)
x <- array_reshape(x, c(1, dim(x)))
x <- imagenet_preprocess_input(x)

system.time({
    features <- model %>% predict(x)
})
# which.max(features)

s = imagenet_decode_predictions(features, top = 1000)[[1]]
tail(s)
print("")
print("")
print("")
print("")

base_model <- application_inception_v3(weights = 'imagenet', include_top = FALSE)

# add our custom layers
predictions <- base_model$output %>% 
    layer_global_average_pooling_2d() %>% 
    layer_dense(units = 1024, activation = 'relu') %>% 
    layer_dense(units = 200, activation = 'softmax')

model <- keras_model(inputs = base_model$input, outputs = predictions)
