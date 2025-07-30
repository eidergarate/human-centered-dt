library(reticulate)
source_python(here::here("src", "auxiliars", "scale_data_controllable_vars.py"), convert = T, envir = globalenv())
library(dplyr)
library(magrittr)



main_scale_data <- function(data_all, data_path_scaler = here::here(Sys.getenv("scaler_fold"), "M1_scaler.pkl")){
  
  data_all %<>% select("") #Variables selection is confidential
  
  load_scaler_and_transform(data_all, data_path_scaler) -> scaled_filtered_data
  return(scaled_filtered_data)
}


