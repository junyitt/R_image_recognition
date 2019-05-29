library(shiny)
library(dplyr)
library(keras)

get_pred <- function(img_path, model, class_indices,  n = 5, img_size = 150){
    img <- image_load(img_path, target_size = c(img_size,img_size))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    
    system.time({
        features <- model$predict(x)
    })
    
    predicted_index <- which.max(features)
    predicted_class <- names(class_indices)[[predicted_index]]
    pred_df <- data.frame(class = names(class_indices), indices = 1:length(features), prob = features[,])
    pdf <- pred_df %>% top_n(5,prob) %>% arrange(-prob) %>% select(Class = class, Probability = prob)
    return(pdf)
}


get_img_path <- function(img_name, img_path = "C:/Users/jy/Desktop/R_IR_7004/DataTest", add_extension = T){
    if(add_extension){
        test_img <- paste0(img_name, ".jpeg")
    }else{
        test_img <- img_name    
    }
    
    final_img_path <- file.path(img_path, test_img)
    if(!file.exists(final_img_path)){
        test_img <- paste0(img_name, ".jpg")
    }
    test_img <- file.path(img_path, test_img)
    return(test_img)
}

