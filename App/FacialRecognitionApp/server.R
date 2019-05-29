library(shiny)
library(dplyr)
library(keras)

print(getwd())
source("../custom_functions.R")

# Parameters --------------------------------------------------------------
path <- "../../"
cache_path <- file.path(path, "Cache/UploadedImages")
model_path <- file.path(path, "Models")
dir.create(cache_path, showWarnings = FALSE)
identity_model_id <- "LFW4"
gender_model_id <- "UTK6_Gender"
age_model_id <- "UTK_Age_3"

# Load Identity Model
load(file = file.path(model_path, paste0("class_indices_", identity_model_id, ".rdata")))
face_class_indices <- class_indices
model <- load_model_hdf5(file.path(model_path, paste0("model_", identity_model_id, ".h5")), compile = F)
print("Loaded Identity Model.")

# Load Gender Model
load(file = file.path(model_path, paste0("class_indices_", gender_model_id, ".rdata")))
gender_class_indices <- class_indices
gender_model <- load_model_hdf5(file.path(model_path, paste0("model_", gender_model_id, ".h5")), compile = F)
print("Loaded Gender Model.")

# Load Age Model
load(file = file.path(model_path, paste0("class_indices_", age_model_id, ".rdata")))
age_class_indices <- class_indices
age_model <- load_model_hdf5(file.path(model_path, paste0("model_", age_model_id, ".h5")), compile = F)
print("Loaded Age Model.")

# Run a sample prediction:
img_path <- "../sample_image.jpg"
pred_df <- get_pred(img_path, model, face_class_indices,  n = 5)
g_pred_df <- get_pred(img_path, gender_model, gender_class_indices,  n = 2, img_size = 128)
a_pred_df <- get_pred(img_path, age_model, age_class_indices,  n = 2, img_size = 128)

server = shinyServer(function(input, output,session){
    output$loading_state <- renderText("")
    observeEvent(input$myFile, {
        inFile <- input$myFile
        if (is.null(inFile))
            return()
        print(inFile)
        name_list <- inFile[,"name"]
        path_list <- inFile[,"datapath"]
        
        d1 <- path_list[1]
        n1 <- name_list[1]
        file.copy(d1, file.path(cache_path, n1))
        
        img_path <- file.path(cache_path, n1)
        print("image path")
        print(img_path)
        
        output$image1 <- renderImage({
            return(list(
                src = img_path,
                contentType = "image",
                width = 300,
                height = 300,
                alt = "Face"
            ))
            
        }, deleteFile = FALSE)
        
        output$image2 <- renderImage({
            return(list(
                src = img_path,
                contentType = "image",
                width = 300,
                height = 300,
                alt = "Face"
            ))
            
        }, deleteFile = FALSE)
        
        output$image3 <- renderImage({
            return(list(
                src = img_path,
                contentType = "image",
                width = 300,
                height = 300,
                alt = "Face"
            ))
        }, deleteFile = FALSE)
        
        print("ok")
        print(gender_class_indices)
        pred_df <- get_pred(img_path, model, face_class_indices,  n = 5)
        g_pred_df <- get_pred(img_path, gender_model, gender_class_indices,  n = 2, img_size = 128)
        a_pred_df <- get_pred(img_path, age_model, age_class_indices,  n = 2, img_size = 128)
        print("predok")
        output$predictTable <- renderTable(pred_df)
        output$predictGenderTable <- renderTable(g_pred_df)
        output$predictAgeTable <- renderTable(a_pred_df)
        
    })
    
    
})