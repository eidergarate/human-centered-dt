
get_last_version <- function(model_path = here::here(Sys.getenv('scaler_fold'))){
  
  all_available_versions <- list.files(model_path)
  
  last_version <- last(sort(all_available_versions))
  
  return(last_version)
}

get_actual_model_path <- function(model_path){
  
  last_version <- get_last_version(model_path)
  
  actual_model_path <- paste0(model_path, "/", last_version)
  
  return(actual_model_path)
}

get_actual_scaler_version <- function(data_path_scaler, model_path){
  
  last_version <- get_last_version(model_path)
  
  last_version_date <- substr(last_version, 17, 26)
  
  actual_scaler_path <- paste0(data_path_scaler, "/M1_scaler_", last_version_date, ".pkl")
  
  return(actual_scaler_path)
}

update_model_version <- function(model_path = here::here(Sys.getenv('scaler_fold'))){
  
  new_date <- gsub("-", "_", Sys.Date())
  
  updated_model_path <- paste0(model_path, "/M1_model_", new_date, ".pkl")
  
  return(updated_model_path)
}

update_scaler_version <- function(data_scaler){
  
  new_date <- gsub("-", "_", Sys.Date())
  
  updated_scaler_path <- paste0(data_scaler, "/M1_model_", new_date, ".pkl")
  
  return(updated_scaler_path)
}

