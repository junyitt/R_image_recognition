library(shiny)
library(dplyr)
library(keras)

source("../custom_functions.R")
options(shiny.maxRequestSize=100*1024^2)

# Parameters --------------------------------------------------------------
path <- "../../"
cache_path <- file.path(path, "Cache/UploadedImages")
model_path <- file.path(path, "Models")
dir.create(cache_path, showWarnings = FALSE)
identity_model_id <- "Identity"
gender_model_id <- "Gender"
age_model_id <- "Age"

unseen_train_data_path <- file.path(path, "Cache/_TRAIN_UNSEEN_LFW")
batch_size <- 32


server = shinyServer(function(input, output,session){
    tryCatch({
    load_time <- system.time({
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
        
        # Person Re-ID intermediate layer
        pretrained_model <- model
        img_size <- pretrained_model$input_shape[[2]]
        
        layer_index <- 157
        inter_layer <- get_layer(pretrained_model, index = layer_index)
        intermediate_layer_model <- keras_model(inputs = pretrained_model$input, outputs = inter_layer$output)
        print("Loaded Person Re-ID layer.")
        
        ##############################################################################################
        # Load the train_features and train_label 
        ##############################################################################################
        load(file = file.path(path, "Cache/train_features.rda"))
        load(file = file.path(path, "Cache/train_label.rda"))
        
        
        # Run a sample prediction:
        img_path <- "../sample_image.jpg"
        pred_df <- get_pred(img_path, model, face_class_indices,  n = 5)
        g_pred_df <- get_pred(img_path, gender_model, gender_class_indices,  n = 2, img_size = 128)
        a_pred_df <- get_pred(img_path, age_model, age_class_indices,  n = 5, img_size = 128)
        sample_features <- get_features(img_path, img_size, intermediate_layer_model)
        print("Done: Sample Prediction.")
    })
    print(load_time)
    
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
        
        pred_df <- get_pred(img_path, model, face_class_indices,  n = 5)
        g_pred_df <- get_pred(img_path, gender_model, gender_class_indices,  n = 2, img_size = 128)
        a_pred_df <- get_pred(img_path, age_model, age_class_indices,  n = 2, img_size = 128)
        output$predictTable <- renderTable(pred_df)
        output$predictGenderTable <- renderTable(g_pred_df)
        output$predictAgeTable <- renderTable(a_pred_df)
    })
        
        labels_reid <- reactiveValues(all_train_features = list(), all_train_label = list(),
                                      unseen_train_label = list())
        observeEvent(input$TrainButton, {
            output$train_loading_state <- renderText("Loading...")
            print("uploading files...")
            TrainFiles <- input$TrainFile
            if (is.null(TrainFiles))
                return()
            path_list <- TrainFiles[,"datapath"]
            N <- length(path_list)
            person_name <- input$TrainClass
            print("upload complete...")
            print(person_name)
            dir.create(file.path(unseen_train_data_path, person_name), showWarnings = F)
            for(each_image in path_list){
                file.copy(each_image, file.path(unseen_train_data_path, person_name))
            }
            
            unseen_train_data_generator <- flow_images_from_directory(unseen_train_data_path,
                                                                      target_size = c(img_size, img_size),
                                                                      shuffle = F
            )
            unseen_train_features <- intermediate_layer_model %>% predict_generator(unseen_train_data_generator, steps = as.integer(unseen_train_data_generator$samples/batch_size))
            unseen_train_label <- get_label_from_generator(unseen_train_data_generator)
            
            ##############################################################################################
            # Combine both train_features and unseen_train_features; train_label and unseen_train_label
            ##############################################################################################
            all_train_features <- rbind(train_features, unseen_train_features)
            all_train_label <- c(train_label, unseen_train_label)
            print("Person Re-ID: Training Done!")
            
            labels_reid$all_train_features <- all_train_features
            labels_reid$all_train_label <- all_train_label
            labels_reid$unseen_train_label <- unseen_train_label
            
            train_class_df <- data.frame(table(unseen_train_features))
            colnames(train_class_df) <- c("Identity", "NumberOfImages")
            output$train_class_table <- renderTable(train_class_df)
            output$train_loading_state <- renderText(paste0("Features extracted from ", N ," images."))
        })
        
        observeEvent(input$TestFile, {
            TestImage <- input$TestFile
            if (is.null(TestImage))
                return()
            img_path <- TestImage[1,"datapath"]
            output$image4 <- renderImage({
                return(list(
                    src = img_path,
                    contentType = "image",
                    width = 300,
                    height = 300,
                    alt = "Face"
                ))
            }, deleteFile = FALSE)
            
            all_train_features <- labels_reid$all_train_features
            all_train_label <- labels_reid$all_train_label 
            unseen_train_label <- labels_reid$unseen_train_label 
            unseen_test_features <- get_features(img_path, img_size, intermediate_layer_model)
            ypred <- KODAMA::knn.kodama(Xtrain = all_train_features, Ytrain = as.factor(all_train_label), Xtest = unseen_test_features, k = 7)$Ypred
            y_pred <- apply(ypred, 1, getmode) # Majority vote on the label will be the final predicted label
            
            freq <- data.frame(table(ypred))
            colnames(freq) <- c("Identity", "LikelihoodScore")
            freq[, "LikelihoodScore"] <- freq[, "LikelihoodScore"]/sum(freq[, "LikelihoodScore"])
            freq <- freq %>% arrange(-LikelihoodScore) %>% as.data.frame()
            output$predictIdentity4Table <- renderTable(freq)
            
            
            
            # Perform t-SNE to visualize the features extracted from the pre-trained model
            nr <- nrow(all_train_features)
            ss <- sample(1:nr, 2000)
            all_features <- rbind(all_train_features[ss,], unseen_test_features)
            all_labels <- c(all_train_label[ss], unseen_test_label)
            
            rstne_result <- Rtsne(all_features, check_duplicates = F)
            tsne_features_df <- data.frame(Feature_1 = rstne_result$Y[,1], Feature_2 = rstne_result$Y[,2], Identity = all_labels) 
            
            train_label_sample <- unique(unseen_train_label)
            train_data_tsne_plot <- tsne_features_df %>% 
                filter(Identity %in% train_label_sample) %>%
                ggplot(aes(x = Feature_1, y = Feature_2, col = Identity)) + 
                geom_point(shape=2) +
                ggtitle("High-Dimensional Features Visualization using t-SNE on Training Data")
            output$features_plot <- renderPlot(train_data_tsne_plot)
            
        })
        
    
    
    }, error = function(e){
        print("Error!")
        print(e)
        session$reload()
    })
    
})