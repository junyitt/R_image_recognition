source("./crop_face.R")

data_path <- "./CustomTest"
filtered_data_path <- "./FILTER_CustomTest"
dir.create(filtered_data_path, showWarnings = F)

file.copy(data_path, filtered_data_path, recursive = T)

img_filenames <- list.files(filtered_data_path, recursive = T, full.names = T)


sapply(img_filenames, FUN = function(img_file){
    output_img <- py$get_crop_img(img_file) # Crop face only
    if(!is.null(output_img)){
        cv2$imwrite(img_file, output_img) # Write the crop image to destination file
    }
    
})