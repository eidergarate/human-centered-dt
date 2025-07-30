library(RMariaDB)
library(logger)

source(here::here("src", "auxiliars", "aux_connect_historicDB.R"))
source(here::here("src", "auxiliars", "aux_optimal_speeds_functions.R"))


SORTING_VARIABLE <- "quality"


select_data_from_db <- function() {

  con <- connect_to_historicDB()
  
  query <- paste0("SELECT ",
                  "Confidential")
  
  result <- dbSendQuery(con, query)
  
  result_df <- dbFetch(result)
  
  dbClearResult(result)
  
  disconnect_from_historicDB(con)
  
    selection <- result_df %>%
    filter(--- == "---") %>% 
    group_by(EX_INF_recipe_ON) %>%
    slice(which.max(.data[[SORTING_VARIABLE]])) %>%
    mutate(Revised = FALSE) %>% 
    select(-c(.data[[SORTING_VARIABLE]], ---)) #Confidential variable
  
  
  return(selection)
  
}



compare_optimal_speeds <- function(new_optimal_speeds) {
  
  optimal_speeds <- read_csv(OPTIMAL_SPEEDS_PATH) %>% 
    mutate(Version = "Old")
  
  new_optimal_speeds <- new_optimal_speeds %>% 
    mutate(Version = "New")
  
  joint_data <- optimal_speeds %>% 
    full_join(new_optimal_speeds)
  
  individual_data <- joint_data %>%
    group_by(EX_INF_recipe_ON) %>% 
    filter(n() == 1) %>% 
    ungroup()
  
  grouped_data <- joint_data %>% 
    group_by(EX_INF_recipe_ON) %>% 
    filter(n() > 1) %>% 
    mutate(group_id = cur_group_id()) %>% 
    ungroup() %>% 
    split(.$group_id) %>% 
    lapply(function(recipes) {
      
      return(recipes[recipes$Version == "New", ])
      
    }) %>% 
    bind_rows() %>% 
    select(-group_id)
  
  updated_optimal_speeds <- rbind(individual_data, grouped_data) %>% 
    select(-Version)
  
  return(updated_optimal_speeds)
  
}



update_optimal_speeds <- function() {
  
  log_info("Getting CONTINENTAL data")
  tryCatch({
    recipe_extract <- load_recipe_data()},
    error = function(e){
      log_error(e)
      log_error("ERROR: Problem getting CONTINENTAL data")
    })
  
  log_info(paste0("Getting data from CONTI_UC2_PROCESSED using ", SORTING_VARIABLE))
  tryCatch({
    selection <- select_data_from_db()},
    error = function(e){
      log_error(e)
      log_error("ERROR: Problem getting data from CONTI_UC2_PROCESSED")
    })
  
  log_info("Updating optimal_speeds.csv")
  tryCatch({
    new_optimal_speeds <- join_data_with_extract(selection, recipe_extract)
    updated_optimal_speeds <- compare_optimal_speeds(new_optimal_speeds) %>% 
      fix_slopes_NA() %>% 
      fix_slopes()
    write_csv(updated_optimal_speeds, OPTIMAL_SPEEDS_PATH)
    log_info("Update successful")},
    error = function(e){
      log_error(e)
      log_error("ERROR: Problem updating optimal_speeds.csv")
    })
  
}
