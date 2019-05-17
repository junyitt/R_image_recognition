library(keras)

# main_path <- "C:/Users/jy/Desktop/R_IR_7004/Archive_Data/testimdb"
main_path <- "C:/Users/jy/Desktop/R_IR_7004/Archive_Data/imdb_crop"
clean_path <- "C:/Users/jy/Desktop/R_IR_7004/Archive_Data/imdb_crop_clean"
filter_clean_path <- "C:/Users/jy/Desktop/R_IR_7004/Archive_Data/imdb_crop_filter"
dir.create(clean_path, showWarnings = F)
dir.create(filter_clean_path, showWarnings = F)

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
    

folders <- list.files(clean_path, full.names = F)
num_photo_threshold <- 50
k <- sapply(folders, FUN = function(j){
    origin <- file.path(clean_path, j)
    if(length(list.files(origin)) > num_photo_threshold){
        file.copy(file.path(clean_path, j), filter_clean_path, recursive=TRUE)
    }
})

})
