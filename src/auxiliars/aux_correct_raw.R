library(magrittr)
library(DBI)
library(dplyr)

locf_leave_front_empty <- function(column){
  
  n <- length(column)
  
  new_col <- zoo::na.locf(column)
  
  n_new <- length(new_col)
  
  new_col <- c(rep(NA, n-n_new), new_col)
  
  return(new_col)
}



correct_data <- function(raw_data, deployment = T){
  
  tryCatch({
    if(!deployment){
      raw_data <- dqts::handleDQ(data = raw_data, metric = 'TimeUniqueness', var_time_name = 'Date', maxdif = 1, units = 'secs', method = 'deletion')
    }
  }, error = function(e){
    print("Data is coming with unique values for each second. This code line is useless.")
  }
  )
  
  tryCatch({
    if(!deployment){
      raw_data <- dqts::handleDQ(data = raw_data, metric = 'Timeliness', var_time_name = 'Date', maxdif = 1, units = 'secs', method = 'missing')
    }
  }, error = function(e){
    print("All columns have 1s frequency values, this code line is useless.")
  }
  )
  
  tryCatch({
    if(!deployment){
      raw_data %<>% mutate_if(purrr::is_character, .funs = locf_leave_front_empty ) %>% mutate_if(purrr::is_numeric, .funs = locf_leave_front_empty)
    }
  }, error = function(e){
    print("There is a problem when inputing data with LOCF method.")
  }
  )
  
  return(raw_data)
}