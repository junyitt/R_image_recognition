source("./Functions/crop_face.R")

data_path <- file.path(getwd(), "Data/lfw-deepfunneled")
filtered_data_path <- file.path(getwd(), "Data/_TRAIN_UNSEEN_LFW")  
dir.create(filtered_data_path, showWarnings = F)

#1. Filter folders with >= 15 images
#2. Crop out the face in the image
#3. Copy into a new folder, with each class having its own folder
system.time({
    
folders <- list.files(data_path, full.names = F)
k <- sapply(folders, FUN = function(person_name){
    origin <- file.path(data_path, person_name)
    dest <- file.path(filtered_data_path, person_name)
    if(length(list.files(origin)) >= 12 & length(list.files(origin)) < 15){
        images <- list.files(origin, full.names = F)   
        dir.create(dest, showWarnings = F) # Create Destination folder
        sapply(images, FUN = function(img_file){
            output_img <- py$get_crop_img(file.path(origin, img_file)) # Crop face only
            if(!is.null(output_img)){
                cv2$imwrite(file.path(dest, img_file), output_img) # Write the crop image to destination file
            }
            
        })
    }
})

})
 
print("Completed: PreProcess_Data_LFW_PersonReID.R")
