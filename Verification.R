
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
model_id <- "A1"
path <- "C:/Users/jy/Desktop/R_IR_7004/"
test_path <- "C:/Users/jy/Desktop/R_IR_7004/DataTestTemp"
model_path <- file.path(path, "Models")

load(file = file.path(model_path, paste0("class_indices_", model_id, ".rdata")))
model <- load_model_hdf5(file.path(model_path, paste0("model_", model_id, ".h5")), compile = F)

test_img_folder <- "C:/Users/jy/Desktop/R_IR_7004/DataTest"
test_data_generator <- flow_images_from_directory(directory = test_img_folder,
                                                  target_size = c(180, 180), batch_size = 32)

model$layers
layer_index <- 157
# layer_index <- 158
inter_layer <- get_layer(model, index = layer_index)
intermediate_layer_model <- keras_model(inputs = model$input, outputs = inter_layer$output)

# 
# img <- image_load(img_path, target_size = c(180,180))
# x <- image_to_array(img)
# x <- array_reshape(x, c(1, dim(x)))
# 
# system.time({
#     features <- intermediate_layer_model$predict(x)
# })
# str(features)

test_img_folder <- "C:/Users/jy/Desktop/R_IR_7004/Data"
test_data_generator <- flow_images_from_directory(directory = test_img_folder,
                                                 target_size = c(180, 180), batch_size = 32, shuffle = F)

intermediate_output <- intermediate_layer_model$predict_generator(test_data_generator)
kk <- sapply(1:dim(intermediate_output)[1], FUN = function(j) c(intermediate_output[j,]))
k1 <- t(kk)
str(k1)

labels <- sapply(strsplit(x = test_data_generator$filenames, split = "[\\]"), FUN = function(x){
    return(x[1])  
})

# install.packages('Rtsne')
library(Rtsne)
library(ggplot2)

## Executing the algorithm on curated data
# tsne <- Rtsne(k2, dims = 3, perplexity=25, verbose=TRUE, max_iter = 2000, check_duplicates = FALSE)
tsne <- Rtsne(k1, check_duplicates = FALSE)

## Plotting
pdf <- data.frame(y1 = tsne$Y[,1], y2 = tsne$Y[,2], label = labels) 
str(pdf)


U <- unique(labels)#[21:30]
U <- U[!is.na(stringr::str_locate(string = U, pattern = "_")[,1])][1:8]
U <- c("_elvispresley", "_junyitt", "_elvispresley2", "_yinyen")
pdf %>% 
    filter(label %in% U) %>%
    ggplot(aes(x = y1, y = y2, col = label)) + geom_point(shape=18) + theme(legend.position = "none")

pdf %>% 
    filter(label %in% U) %>%
    ggplot(aes(x = y1, y = y2, col = label)) + geom_point(shape=18) #+ theme(legend.position = "none")


# zz <- apply(k1, 1, FUN = function(j) which.max(j))
# apply(intermediate_output, 1, FUN = function(j) sum(j))
# zdf <- data.frame(pred = zz, truth = 1 + (test_data_generator$classes))
# head(zdf)
# 
# mean(zdf$pred == zdf$truth)
