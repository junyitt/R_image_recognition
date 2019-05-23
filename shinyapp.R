library(shiny)
library(dplyr)
library(keras)

get_pred <- function(img_path, model, class_indices,  n = 5, img_size = 150){
    img <- image_load(img_path, target_size = c(img_size,img_size))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    
    system.time({
        features <- model$predict(x)
    })
    
    # class_indices <- data_generator$class_indices
    predicted_index <- which.max(features)
    predicted_class <- names(class_indices)[[predicted_index]]
    pred_df <- data.frame(class = names(class_indices), indices = 1:length(features), prob = features[,])
    pdf <- pred_df %>% top_n(5,prob) %>% arrange(-prob)
    return(pdf)
}


get_img_path <- function(img_name, img_path = "C:/Users/jy/Desktop/R_IR_7004/DataTest", add_extension = T){
    if(add_extension){
        test_img <- paste0(img_name, ".jpeg")
    }else{
        test_img <- img_name    
    }
    
    final_img_path <- file.path(img_path, test_img)
    if(!file.exists(final_img_path)){
        test_img <- paste0(img_name, ".jpg")
    }
    test_img <- file.path(img_path, test_img)
    return(test_img)
}



# Parameters --------------------------------------------------------------
model_id <- "LFW2"
path <- "."
test_path <- "./DataTestTemp"
model_path <- file.path(path, "Models")

load(file = file.path(model_path, paste0("class_indices_", model_id, ".rdata")))
face_class_indices <- class_indices
model <- load_model_hdf5(file.path(model_path, paste0("model_", model_id, ".h5")), compile = F)

# Gender
gender_model_id <- "UTK6_Gender"
load(file = file.path(model_path, paste0("class_indices_", gender_model_id, ".rdata")))
gender_class_indices <- class_indices
gender_model <- load_model_hdf5(file.path(model_path, paste0("model_", gender_model_id, ".h5")), compile = F)


# Age
age_model_id <- "UTK_Age_3"
load(file = file.path(model_path, paste0("class_indices_", age_model_id, ".rdata")))
age_class_indices <- class_indices
age_model <- load_model_hdf5(file.path(model_path, paste0("model_", age_model_id, ".h5")), compile = F)

shinyApp(
    ui = shinyUI(  
        fluidRow( 
            h1("Select Input"),
            fileInput("myFile", "Choose an image file:", 
                      accept = c('image/png', 'image/jpeg', 'image/jpg'),
                      multiple = T
                      ),
            h1("Input Image"),
            imageOutput("image2", width = "300px", height = "300px"),
            h1("Predicted Class"),
            tableOutput('predictTable'),
            tableOutput('predictGenderTable'),
            tableOutput('predictAgeTable')
            
        )
    ),
    server = shinyServer(function(input, output,session){
        observeEvent(input$myFile, {
            inFile <- input$myFile
            if (is.null(inFile))
                return()
            print(inFile)
            name_list <- inFile[,"name"]
            path_list <- inFile[,"datapath"]
            
            d1 <- path_list[1]
            n1 <- name_list[1]
            file.copy(d1, file.path(test_path, n1))
            
            img_path <- tryCatch({
                img_path <- get_img_path(inFile$name, img_path = test_path, add_extension = F)
                img_path
            })
            
            output$image2 <- renderImage({
                return(list(
                    src = img_path,
                    contentType = "image",
                    width = 300,
                    height = 300,
                    alt = "Face"
                ))
                
            }, deleteFile = FALSE)
            
            
            pred_df <- get_pred(img_path, model, face_class_indices,  n = 5)
            g_pred_df <- get_pred(img_path, gender_model, gender_class_indices,  n = 2, img_size = 128)
            a_pred_df <- get_pred(img_path, age_model, age_class_indices,  n = 2, img_size = 128)
            output$predictTable <- renderTable(pred_df)
            output$predictGenderTable <- renderTable(g_pred_df)
            output$predictAgeTable <- renderTable(a_pred_df)
            
        })
        
        
    })
)

# shinyApp(ui, server)