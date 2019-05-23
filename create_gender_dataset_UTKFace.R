data_path <- "./UTKFace/UTKFace"
gender_data_path <- "./FILTER_Gender_UTKFace"
dir.create(gender_data_path, showWarnings = F)


img_files <- list.files(data_path)
status <- sapply(img_files, FUN = function(img_file){
    age <- strsplit(img_file, "[_]")[[1]][1]
    gender <- strsplit(img_file, "[_]")[[1]][2]
    ethnic <- strsplit(img_file, "[_]")[[1]][3]
    
    orig_path <- file.path(data_path, img_file)
    gender_dest_path <- file.path(gender_data_path, gender)
    dir.create(gender_dest_path, showWarnings = F)
    
    file.copy(orig_path, gender_dest_path)
})



