library(dplyr) #For pipe and data manipulation
library(stringr)#for string manipulation 
library(purrr)# for effective itteration

Extract_ModelData <- function(files){
  
  name_files <- files %>% 
    basename() #Gets file name
  
  
  files <- files %>% 
    set_names(name_files)#Used in map so each row has an file name ID
  
  
  df_results <- map_dfr(files, read.csv, .id = "File_Name") #From purr
  
  # Data processing -----
  df_results <- df_results %>% 
    mutate(Temp = str_split(File_Name, "_")) %>% 
    mutate(Model = map_chr(Temp, 1), #Name of model
           Dataset = map_chr(Temp,2), #CC or MOOC
           Type = map_chr(Temp,3), # TT or sub
           Job = str_sub(map_chr(Temp,4), start = 1, end = 7)) %>% #Job ID from HPC
    select(-Temp) %>% 
    group_by(Job) %>% #Group by Job so we can assign batch rows within each job
    mutate(Batch = seq(n())) %>% #numbering the rows by batch size we assume ordered already
    ungroup()
  
  df_results
}

Summarise_ModelResults <- function(dataframe){
  df_summary <- dataframe %>% 
    group_by(Dataset, Model, Type, Batch) %>%
    summarise(across(AP:F1,#Variables of interest
                     list(mean = ~mean(.x, na.rm = T), sd = sd,
                          lower = ~mean(.x) - sd(.x),#Used in plotting
                          higher = ~mean(.x) + sd(.x)), 
                     .names = "{.col}_{.fn}")) %>%  #Name of the columns
    ungroup() %>% 
    mutate(Model_Type = str_c(Model,"_", Type))
  
  df_summary
}
