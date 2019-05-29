train_fdn_path <- file.path(getwd(), "Data/_TRAIN_faces-data-new")
test_fdn_path <- file.path(getwd(), "Data/_TEST_faces-data-new")  

train_lfw_path <- file.path(getwd(), "Data/_TRAIN_lfw-deepfunneled")
test_lfw_path <- file.path(getwd(), "Data/_TEST_lfw-deepfunneled")  

train_combined_path <- file.path(getwd(), "Data/_TRAIN_Combined_IdentityData")
test_combined_path <- file.path(getwd(), "Data/_TEST_Combined_IdentityData")
dir.create(train_combined_path, showWarnings = F)
dir.create(test_combined_path, showWarnings = F)

# Combine: Copy both Training and Testing LFW and FDN data to a Training folder and a Testing folder respectively.
copy_status <- sapply(list.files(train_fdn_path, full.names = T), FUN = function(j){
    file.copy(j, train_combined_path, recursive = T)
})

copy_status <- sapply(list.files(test_fdn_path, full.names = T), FUN = function(j){
    file.copy(j, test_combined_path, recursive = T)
})

copy_status <- sapply(list.files(train_lfw_path, full.names = T), FUN = function(j){
    file.copy(j, train_combined_path, recursive = T)
})

copy_status <- sapply(list.files(test_lfw_path, full.names = T), FUN = function(j){
    file.copy(j, test_combined_path, recursive = T)
})

print("Completed: PreProcess_Combine_LFW_FDN.R")


