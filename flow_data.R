library(keras)

# Parameters --------------------------------------------------------------

batch_size <- 32
epochs <- 30

# Data Preparation --------------------------------------------------------
image_data_generator_1 <- image_data_generator(
    rotation_range = 45, #20
    width_shift_range = 0.4, #0.2
    height_shift_range = 0.4, #0.2
    horizontal_flip = TRUE, 
    validation_split = 0.2
)
# data_generator <- flow_images_from_directory("C:/Users/jy/Desktop/R_IR_7004/faces-data-new/clean_train", 
#                                              generator = image_data_generator_1,
#                                              target_size = c(180, 180)
#                                             )
# val_data_generator <- flow_images_from_directory("C:/Users/jy/Desktop/R_IR_7004/faces-data-new/clean_val", 
#                                              generator = image_data_generator_1,
#                                              target_size = c(180, 180)
# )

data_generator <- flow_images_from_directory("C:/Users/jy/Desktop/R_IR_7004/faces-data-new/main_data", 
                                             generator = image_data_generator_1,
                                             target_size = c(180, 180)
)


input_img <- layer_input(shape = c(180, 180, 3))
num_classes <- data_generator$num_classes

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
        data_generator,
        steps_per_epoch = as.integer(50000/batch_size), 
        epochs = epochs,
        validation_data = val_data_generator
        # validation_data = flow_images_from_data(x_test, y_test, datagen, batch_size = batch_size)
    )
})


model %>% save_model_hdf5("C:/Users/jy/Desktop/R_IR_7004/test_model_1.h5")

# library(dplyr)
# get_pred <- function(img_path, n = 5){
#     img <- image_load(img_path, target_size = c(180,180))
#     x <- image_to_array(img)
#     x <- array_reshape(x, c(1, dim(x)))
#     
#     system.time({
#         features <- model %>% predict(x)
#     })
#     
#     class_indices <- data_generator$class_indices
#     predicted_index <- which.max(features)
#     predicted_class <- names(class_indices)[[predicted_index]]
#     pred_df <- data.frame(class = names(class_indices), indices = 1:length(features), prob = features[,])
#     pdf <- pred_df %>% top_n(5,prob) %>% arrange(-prob)
#     return(pdf)
# }
# # Predict
# 
# get_img_path <- function(img_name, img_path = "C:/Users/jy/Desktop/test"){
#     test_img <- paste0(img_name, ".jpeg")
#     final_img_path <- file.path(img_path, test_img)  
#     if(!file.exists(final_img_path)){
#         test_img <- paste0(img_name, ".jpg")
#     }
#     test_img <- file.path(img_path, test_img)    
#     return(test_img)
# }
# 
# 
# img_path <- get_img_path("WhatsApp Image 2019-05-05 at 18.19.15(6) - Copy")
# get_pred(img_path)


