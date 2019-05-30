data_path <- file.path(getwd(), "Data/_TRAIN_UNSEEN_LFW")
test_data_path <- file.path(getwd(), "Data/_TEST_UNSEEN_LFW") 
dir.create(test_data_path, showWarnings = F)

folders <- list.files(data_path, full.names = F)
test_set_proportion = 0.20

set.seed(123)
#1. For each class, select 20% of the images, and move to a separate TEST directory (as test dataset)
status <- sapply(folders, FUN = function(person_name){
    origin <- file.path(data_path, person_name)
    dest <- file.path(test_data_path, person_name)
    dir.create(dest, showWarnings = F)
    
    img_files <- list.files(origin)
    num_test_samples <- as.integer(length(img_files)*test_set_proportion) # Select 20% of the image of that particular class as test dataset
    test_img_files <- sample(img_files, size = num_test_samples)
    for(img_file in test_img_files){
        file.rename(from = file.path(origin, img_file), to = file.path(dest, img_file))
    }
})



