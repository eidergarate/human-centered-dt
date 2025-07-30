library(logger)

source(here::here("src", "auxiliars", "aux_optimal_speeds_functions.R"))

HISTORICAL_DATA_PATH <- here::here("preprocessed_data_MW1.csv")



load_historical_data <- function() {
  
  historical_data <- read_csv(HISTORICAL_DATA_PATH) %>% 
    filter(--- == "---") %>% 
    select(---) #Confidential filtering and selection
  
  historical_selection <- historical_data %>%
    group_by(recipe) %>%
    slice(which.max(quality)) %>%
    select(-quality) %>% 
    mutate(Revised = FALSE)
  
  return(historical_selection)
  
}



get_historical_optimal_speeds <- function() {
  
  log_info("Getting CONTINENTAL data")
  tryCatch({
    recipe_extract <- load_recipe_data()},
    error = function(e){
      log_error(e)
      log_error("ERROR: Problem getting CONTINENTAL data")
    })
  
  log_info("Getting historical data")
  tryCatch({
    historical_selection <- load_historical_data()},
    error = function(e){
      log_error(e)
      log_error("ERROR: Problem getting historical data")
    })
  
  log_info("Creating initial optimal_speeds.csv")
  tryCatch({
    optimal_speeds <- join_data_with_extract(historical_selection, recipe_extract) %>% 
      fix_slopes_NA() %>% 
      fix_slopes()
    write_csv(optimal_speeds, OPTIMAL_SPEEDS_PATH)
    log_info("Creation sucessful")},
    error = function(e){
      log_error(e)
      log_error("ERROR: Problem creating initial optimal_speeds.csv")
    })
  
  
}
