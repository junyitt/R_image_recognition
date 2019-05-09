library(shiny)

library(dplyr)
library(keras)

get_pred <- function(img_path, model, class_indices,  n = 5){
    img <- image_load(img_path, target_size = c(180,180))
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
model_id <- "4"
path <- "C:/Users/jy/Desktop/R_IR_7004/"
test_path <- "C:/Users/jy/Desktop/R_IR_7004/DataTestTemp"
model_path <- file.path(path, "Models")

load(file = file.path(model_path, paste0("class_indices_", model_id, ".rdata")))
model <- load_model_hdf5(file.path(model_path, paste0("model_", model_id, ".h5")), compile = F)

test_img_folder <- "C:/Users/jy/Desktop/R_IR_7004/DataTest2/archive"
# img_path <- get_img_path("19274921_1355879227815089_872306674373643628_n", img_path = test_img_folder)
# get_pred(img_path, model, class_indices,  n = 5)




shinyApp(
    ui = shinyUI(  
        fluidRow( 
            fileInput("myFile", "Choose a file", accept = c('image/png', 'image/jpeg', 'image/jpg')),
            column(12,
                tableOutput('predictTable')
            )
        )
    ),
    server = shinyServer(function(input, output,session){
        observeEvent(input$myFile, {
            inFile <- input$myFile
            if (is.null(inFile))
                return()
            file.copy(inFile$datapath, file.path(test_path, inFile$name) )
            img_path <- tryCatch({
                img_path <- get_img_path(inFile$name, img_path = test_path, add_extension = F)
                img_path
            })
            
            pred_df <- get_pred(img_path, model, class_indices,  n = 5)
            output$predictTable <- renderTable(pred_df)
            
        })
        
        
    })
)

# shinyApp(ui, server)