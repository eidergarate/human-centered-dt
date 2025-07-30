library(dplyr)
library(magrittr)
source(here::here("src", "auxiliars", "aux_connect_historicDB.R"))




get_new_name_for_old <- function(){
  
  
  all_files <- list.files(here::here("Data/models_info"))
  
  old_speeds_files <- sort(all_files[grepl("optimal_speeds_old", all_files)])
  last_old <- old_speeds_files[length(old_speeds_files)]
  
  new_id <- as.numeric(substring(strsplit(last_old, "_")[[1]][4], 1, 1)) + 1
  
  new_name_for_old <- sprintf("optimal_speeds_old_%s.csv", new_id)
  
  return(new_name_for_old)
  
}


filter_for_optim_speeds <- function(historical_table){
  
  historical_table %<>% filter(--- == "---") %>% filter(quality_status == 0) #Confidential filter
  
  return(historical_table)
}


select_vars_for_optim <- function(historical_table){
  
  historical_table %<>% select(recipe, ---) #Confidential selection
  
  return(historical_table)
}


get_optimals <- function(historical_table){
  
  historical_table %>% 
    group_by(recipe) %>%
    mutate(---) -> speeds_by_recipe #Confidential
  
  return(speeds_by_recipe)
}




main_update_speed_suggestions <- function(host, port, 
                                          user, password, 
                                          db_name, table_name,
                                          old_optimal_speeds, optimization_launched, 
                                          optimized_results = NULL){
  
  log_info("Renaming last optimal values")
  
  olds_name_after_execution <- get_new_name_for_old()
  
  old_optimal_speeds_table <- read.csv(old_optimal_speeds, stringsAsFactors = F) %>% select(-c(X)) 
  
  write.csv(old_optimal_speeds_table, file = here::here("Data/models_info", olds_name_after_execution))
  
  
  if(!optimization_launched){
    
    optimized_results <- old_optimal_speeds_table %>% filter(recipe %in% c("A", "B", "C"))
  }
  
  log_info("Starting update calculation")
  
  historical_table <- query_historicalDB_preprocessed_all()
  
  historical_table %<>% filter_for_optim_speeds(.)
  
  historical_table %<>% select_vars_for_optim(.)
  
  optimal_new_values %<>% get_optimals(.)
  
  optimal_new_values %<>% filter(!recipe %in% c("A", "B", "C"))
  
  optimal_new_values <- rbind(optimal_new_values, optimized_results)
  
  write.csv(optimal_new_values, file = here::here("Data/models_info/optimal_speeds.csv"))
  
  log_info("Succesfully updated")
  
  return(True)
  
}