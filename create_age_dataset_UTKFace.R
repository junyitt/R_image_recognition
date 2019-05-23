data_path <- "./UTKFace/UTKFace"
age_data_path <- "./FILTER_AgeGroup_UTKFace"
dir.create(age_data_path, showWarnings = F)


img_files <- list.files(data_path)
status <- sapply(img_files, FUN = function(img_file){
    age <- strsplit(img_file, "[_]")[[1]][1]
    gender <- strsplit(img_file, "[_]")[[1]][2]
    ethnic <- strsplit(img_file, "[_]")[[1]][3]
    
    age_n <- as.integer(age)
    if(age_n <= 3){
        age_group <- "0-3"
    }else if(age_n <= 6){
        age_group <- "4-6"
    }else if(age_n <= 12){
        age_group <- "7-12"
    }else if(age_n <= 18){
        age_group <- "13-18"
    }else if(age_n <= 25){
        age_group <- "19-25"
    }else if(age_n <= 35){
        age_group <- "26-35"
    }else if(age_n <= 45){
        age_group <- "36-45"
    }else if(age_n <= 60){
        age_group <- "46-60"
    }else if(age_n <= 75){
        age_group <- "61-75"
    }else if(age_n > 75){
        age_group <- "75-"
    }
    
    orig_path <- file.path(data_path, img_file)
    age_dest_path <- file.path(age_data_path, age_group)
    dir.create(age_dest_path, showWarnings = F)
    
    file.copy(orig_path, age_dest_path)
})



