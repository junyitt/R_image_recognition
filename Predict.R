
library(dplyr)
library(keras)

get_pred <- function(img_path, model, class_indices,  n = 5){
    img <- image_load(img_path, target_size = c(180,180))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    
    system.time({
        features <- model$predict(x)
    })
    
    # class_indices <- data_generator$class_indices
    predicted_index <- which.max(features)
    predicted_class <- names(class_indices)[[predicted_index]]
    pred_df <- data.frame(class = names(class_indices), indices = 1:length(features), prob = features[,])
    pdf <- pred_df %>% top_n(5,prob) %>% arrange(-prob)
    return(pdf)
}

get_img_path <- function(img_name, img_path = "C:/Users/jy/Desktop/test"){
    test_img <- paste0(img_name, ".jpeg")
    final_img_path <- file.path(img_path, test_img)
    if(!file.exists(final_img_path)){
        test_img <- paste0(img_name, ".jpg")
    }
    test_img <- file.path(img_path, test_img)
    return(test_img)
}

model_id <- "1"
model_path <- "C:/Users/jy/Desktop/R_IR_7004/Models/"

model <- load_model_hdf5(paste0(model_path, "model_", model_id, ".h5"), compile = F)
load(file = paste0(model_path, "class_indices_", model_id, ".rdata"))
img_path <- get_img_path("boylee.17")
get_pred(img_path, model, class_indices,  n = 5)
