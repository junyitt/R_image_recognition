source("./crop_face.R")

data_path <- "./faces-data-new/images"
filtered_data_path <- "./FILTER_faces-data-new"
dir.create(filtered_data_path, showWarnings = F)

img_filenames <- list.files(data_path)

#1. Crop out the face in the image
#2. Copy into a new folder, with each class having its own folder
status <- sapply(img_filenames, FUN = function(img_filename){

    #Original Image Path
    orig_path <- file.path(data_path, img_filename)
    
    # Get Category/Class Name
    k <- stringr::str_locate(string = img_filename, "[.]")[[1]]
    category_name <- substr(img_filename, 1, k - 1)
    
    # Create dest_path folder
    dest_path <- file.path(filtered_data_path, category_name)
    if(!dir.exists(dest_path)){
        dir.create(dest_path)
    }
    
    # Copy 
    dest_image_path <- file.path(dest_path, img_filename)
    
    output_img <- py$get_crop_img(orig_path) # Crop face only
    if(!is.null(output_img)){
        cv2$imwrite(dest_image_path, output_img) # Write the crop image to destination file
    }
    
})




