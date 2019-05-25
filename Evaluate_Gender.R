library(keras)
library(caret)

evaluate_accuracy <- function(pred, truth){
    pred_fact <- apply(pred, 1, FUN =  function(x) which.max(x)-1)
    return(mean(pred_fact == truth))
}

get_confusion_matrix_table <- function(pred, truth, percentage = T){
    pred_fact <- apply(pred, 1, FUN =  function(x) which.max(x)-1)
    
    p <- factor(pred_fact)
    t <- factor(truth)
    cm <- confusionMatrix(p, t)$table
    attr(cm, "dimnames")$Prediction <- names(test_data_generator$class_indices)
    attr(cm, "dimnames")$Reference <- names(test_data_generator$class_indices)
    if(percentage){
        cm <-cm/rowSums(cm)
        cm <- round(cm*100,2)
        return(cm)
    }else{
        return(cm)
    }
}


# Parameters --------------------------------------------------------------
model_id <- "UTK6_Gender"
path <- "./"
data_path <- file.path(path, "FILTER_Gender_UTKFace")
test_data_path <- file.path(path, "TEST_FILTER_Gender_UTKFace")
model_path <- file.path(path, "Models")
checkpoint_dir <- file.path(path, "Checkpoints")
input_shape_len <- 128
batch_size <- 32

# Load Model --------------------------------------------------------------
load(file = file.path(model_path, paste0("class_indices_", model_id, ".rdata")))
model <- load_model_hdf5(file.path(model_path, paste0("model_", model_id, ".h5")), compile = F)

# Data Preparation --------------------------------------------------------
image_data_generator_1 <- image_data_generator(
    validation_split = 0.2
)

train_data_generator <- flow_images_from_directory(data_path, 
                                                     generator = image_data_generator_1,
                                                     target_size = c(input_shape_len, input_shape_len),
                                                     subset = "training",
                                                     shuffle = F
)

val_data_generator <- flow_images_from_directory(data_path, 
                                                   generator = image_data_generator_1,
                                                   target_size = c(input_shape_len, input_shape_len), 
                                                   subset = "validation",
                                                   shuffle = F
)

test_data_generator <- flow_images_from_directory(test_data_path, 
                                                 generator = image_data_generator_1,
                                                 target_size = c(input_shape_len, input_shape_len), 
                                                 shuffle = F
)

# Predictions --------------------------------------------------------------
train_pred <- model %>% predict_generator(train_data_generator, steps = as.integer(train_data_generator$samples/batch_size))
val_pred <- model %>% predict_generator(val_data_generator, steps = as.integer(val_data_generator$samples/batch_size))
test_pred <- model %>% predict_generator(test_data_generator, steps = as.integer(test_data_generator$samples/batch_size))


train_accuracy <- evaluate_accuracy(train_pred, train_data_generator$labels) # Train Accuracy
val_accuracy <- evaluate_accuracy(val_pred, val_data_generator$labels) # Validation Accuracy
test_accuracy <- evaluate_accuracy(test_pred, test_data_generator$labels) # Test Accuracy
print(paste0("Accuracy (Train): ", train_accuracy))
print(paste0("Accuracy (Validation): ", val_accuracy))
print(paste0("Accuracy (Test): ", test_accuracy))

test_cm <- get_confusion_matrix_table(test_pred, test_data_generator$labels) # Test Confusion Matrix
print("Confusion Matrix (Test):")
print(test_cm)
