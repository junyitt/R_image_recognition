library(keras)

load_and_resize <- function(img_path){
    # img_path <- file.path("C:/Users/jy/Desktop/R_image_recognition/face-images-13233/Images/", img)
    img_pil <- image_load(img_path)
    # p <- image_to_array(img_pil, data_format = c("channels_last", "channels_first"))
    p2 <- image_array_resize(img_pil, 180, 180, data_format = c("channels_last", "channels_first"))
    
    return(p2)
}

main_path <- "C:/Users/jy/Desktop/R_IR_7004/face-images-13233/Images"
clean_path <- "C:/Users/jy/Desktop/R_IR_7004/face-images-13233/clean_1"
img_filenames <- list.files(main_path)

for(img_filename in img_filenames){
    #Original Image Path
    orig_path <- file.path(main_path, img_filename)
    
    # Get Category/Class Name
    k <- stringr::str_locate(string = img_filename, "[_][0-9]")[[1]]
    category_name <- substr(img_filename, 1, k - 1)
    
    # Create dest_path folder
    dest_path <- file.path(clean_path, category_name)
    if(!dir.exists(dest_path)){
        dir.create(dest_path)
    }
    
    # Copy 
    dest_image_path <- file.path(dest_path, img_filename)
    img_pil <- load_and_resize(orig_path)
    image_array_save(img_pil, dest_image_path, data_format = NULL, file_format = NULL, scale = TRUE)
    
}





# 
# 
# 
# cifar10 <- dataset_cifar10()
# k = cifar10$train$x
# dim(k)
# k[1,,,1]
