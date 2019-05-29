library(shiny)
library(shinycssloaders)
library(dplyr)

ui = shinyUI(  
    fluidPage( 
        h1("Facial Recognition App"),
        h3("Upload Input"),
        fileInput("myFile", "Choose an image file:", 
                  accept = c('image/png', 'image/jpeg', 'image/jpg'),
                  multiple = T
        ),
    
        tabsetPanel(type = "tabs",
                    tabPanel("Identity",
                             textOutput("loading_state")  %>% withSpinner(color="#0dc5c1"),
                             imageOutput("image1", width = "300px", height = "300px"),
                             h4("Prediction"),
                             tableOutput('predictTable')
                             ),
                    tabPanel("Gender",
                             imageOutput("image2", width = "300px", height = "300px"),
                             p("0: Male, 1: Female"),
                             h4("Prediction"),
                             tableOutput('predictGenderTable')
                    ),
                    tabPanel("Age",
                             imageOutput("image3", width = "300px", height = "300px"),
                             h4("Prediction"),
                             tableOutput('predictAgeTable')
                    ),
                    tabPanel("Combined",
                             imageOutput("image4", width = "300px", height = "300px") 
                             # tableOutput('predictAgeTable')
                    )
        )
    )
)