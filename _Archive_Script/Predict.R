
library(dplyr)
library(keras)

get_pred <- function(img_path, model, class_indices,  n = 5, img_size = 150){
    img <- image_load(img_path, target_size = c(img_size,img_size))
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


# Parameters --------------------------------------------------------------
model_id <- "LFW2"
path <- "C:/Users/jy/Desktop/R_IR_7004/"
test_path <- "C:/Users/jy/Desktop/R_IR_7004/DataTest"
model_path <- file.path(path, "Models")

load(file = file.path(model_path, paste0("class_indices_", model_id, ".rdata")))
model <- load_model_hdf5(file.path(model_path, paste0("model_", model_id, ".h5")), compile = F)

test_img_folder <- "C:/Users/jy/Desktop/R_IR_7004/DataTest/_elvispresley3"
img_path <- get_img_path("0_PROD-Photo-of-Elvis-PRESLEY", img_path = test_img_folder)
img_path <- get_img_path("elvis_presley_hero_74290646", img_path = test_img_folder)
img_path <- get_img_path("elvis-presley", img_path = test_img_folder)
img_path <- get_img_path("ep2", img_path = test_img_folder)
get_pred(img_path, model, class_indices,  n = 5)

