# source("./crop_face.R")
# data_path <- "./FILTER_Gender_UTKFace"
data_path <- "./FILTER_AgeGroup_UTKFace"
data_path <- "./FILTER_UNSEEN_LFW"
# test_data_path <- "./TEST_FILTER_Gender_UTKFace"
test_data_path <- "./TEST_FILTER_UNSEEN_LFW"
dir.create(test_data_path, showWarnings = F)

folders <- list.files(data_path, full.names = F)

set.seed(123)
#1. For each class, select 10% of the images, and move to a separate TEST directory (as test dataset)
status <- sapply(folders, FUN = function(person_name){
    origin <- file.path(data_path, person_name)
    dest <- file.path(test_data_path, person_name)
    dir.create(dest, showWarnings = F)
    
    img_files <- list.files(origin)
    num_test_samples <- as.integer(length(img_files)*0.2) # Select 10% of the image of that particular class as test dataset
    test_img_files <- sample(img_files, size = num_test_samples)
    for(img_file in test_img_files){
        file.rename(from = file.path(origin, img_file), to = file.path(dest, img_file))
    }
})



