
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

get_img_path <- function(img_name, img_path = "C:/Users/jy/Desktop/R_IR_7004/DataTest"){
    test_img <- paste0(img_name, ".jpeg")
    final_img_path <- file.path(img_path, test_img)
    if(!file.exists(final_img_path)){
        test_img <- paste0(img_name, ".jpg")
    }
    test_img <- file.path(img_path, test_img)
    return(test_img)
}


# Parameters --------------------------------------------------------------
model_id <- "3"
path <- "C:/Users/jy/Desktop/R_IR_7004/"
model_path <- file.path(path, "Models")

load(file = file.path(model_path, paste0("class_indices_", model_id, ".rdata")))
model <- load_model_hdf5(file.path(model_path, paste0("model_", model_id, ".h5")), compile = F)

img_path <- get_img_path("cipa")
get_pred(img_path, model, class_indices,  n = 5)

