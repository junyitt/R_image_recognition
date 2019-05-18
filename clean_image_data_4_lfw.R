source("./crop_face.R")

data_path <- "C:/Users/jy/Desktop/R_IR_7004/Data_lfw-deepfunneled"
filtered_data_path <- "C:/Users/jy/Desktop/R_IR_7004/Data_lfw_filter"
dir.create(filtered_data_path, showWarnings = F)

# 
# 
# all_jpg <- list.files(path = main_path, pattern = "*.jpg", full.names = T,recursive = T)
# 
# for(jpg in all_jpg){
#     split_output <- strsplit(jpg, split = "[/]")[[1]]
#     f <- split_output[length(split_output)]
#     class <- strsplit(f, split = "[_]")[[1]][1]
#     
#     new_class_folder <- file.path(clean_path, class)
#     dir.create(new_class_folder, showWarnings = F)
#     
#     dest_jpg <- file.path(new_class_folder, f)
#     file.copy(from = jpg, to = dest_jpg)
# }


#Filter folders with >50imgs only

system.time({
    
folders <- list.files(data_path, full.names = F)
num_photo_threshold <- 15
k <- sapply(folders, FUN = function(j){
    origin <- file.path(data_path, j)
    if(length(list.files(origin)) > num_photo_threshold){
        file.copy(file.path(data_path, j), filtered_data_path, recursive=TRUE)
    }
})

})

output_img <- py$get_crop_img("Data/9326871/9326871.3.jpg")
cv2$imwrite('output.jpg',output_img)
