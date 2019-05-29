library(reticulate)

py_run_file("./Functions/crop.py")
cv2 <- import("cv2")

# # Example:
# output_img <- py$get_crop_img("Data/Example.jpg")
# cv2$imwrite('output.jpg',output_img)
