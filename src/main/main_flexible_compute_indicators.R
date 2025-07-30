library(dplyr)
library(magrittr)
library(reticulate)
library(DBI)
library(logger)
httr::set_config(httr::config(ssl_verifypeer=0,ssl_verifyhost=0))
source(here::here("src", "main", "main_compute_daily_indicators.R"))
source(here::here("src", "auxiliars", 'aux_connect_production_SQL.R'))
source(here::here("src", "auxiliars", 'aux_connect_historicDB.R'))

last_update_date <- function(){
  
  hist_con <- connect_to_historicDB()
  
  last_usage_query <- paste0("Confidential_query", Sys.getenv("table_preprocessed"), ");")
  
  last_time <- RJDBC::dbGetQuery(hist_con, last_usage_query)
  
  disconnect_from_historicDB(hist_con)
  
  return(lubridate::as_datetime(last_time$starting_time[1]))
}



main_flexible_compute_indicators <- function(now_time = Sys.time(), usage_host, 
                                             usage_port,
                                             usage_dbname,
                                             usage_user,
                                             usage_password){
  

  log_info("Accessing Historical Data to see if there has ben an update")
  
  tryCatch({
      last_updating_time <- last_update_date() + 1
      time_change <- difftime(lubridate::as_datetime(now_time), lubridate::as_datetime(last_updating_time), units = "hours")
      
      if(time_change > 3){
        
        find_first <- T
        
      }else{
        
        find_first <- F
        
      }
    }, error = function(e){
      log_error(e)
    })
  
  
 
  if(!find_first){
    
    log_info("Calculating last datetime for the update")
    
    tryCatch({
      
      from_date <- lubridate::as_datetime(last_updating_time)
      from_date <- sprintf("%sT%sZ", lubridate::date(from_date),  hms::as_hms(from_date))
      
    }, error = function(e){
      log_error(e)
    })
    
    log_info("Accessing Usage table for last usage value")
    tryCatch({
      con_production_SQL <- connect_to_production_usage_SQL()
      usage_query <- sprintf("Confidential query", Sys.getenv("table_usage"))
	    last_usage_date <- RJDBC::dbGetQuery(con_production_SQL, usage_query)
	    last_usage_date %<>% filter(Usage == '1')
	    disconnect_from_production_SQL(con_production_SQL)
    }, error = function(e){
      log_error(e)
      log_error("Last usage value impossible to query")
    })

    last_usage_date <- lubridate::as_datetime(last_usage_date$Date)
    to_date <- sprintf("%sT%sZ", lubridate::date(last_usage_date),  hms::as_hms(last_usage_date))
  }
  else{
    
    log_info("Calculating first and last datetime for the update")
    
    tryCatch({
      
      con_production_SQL <- connect_to_production_usage_SQL()
      usage_query <- sprintf("SELECT * FROM %s WHERE (Date > '%s' AND Date <= '%s');", Sys.getenv("table_usage"), lubridate::as_datetime(now_time - 3*3600), lubridate::as_datetime(now_time))
      usage_last3h <- RJDBC::dbGetQuery(con_production_SQL, usage_query)
      disconnect_from_production_SQL(con_production_SQL)},
      
      error = function(e){
        log_error(e)
      })
    
    
    tryCatch({
      
      first_part <- usage_last3h %>% filter(Date < lubridate::as_datetime(now_time - 1.5*3600) & Usage == 1) %>% select(Date)
    
      time_2hago <- lubridate::as_datetime(now_time - 2*3600)
    
      min_diff_first <- which.min(abs(difftime(time_2hago, first_part[ ,1])))
    
      from_date <- first_part[min_diff_first, 1]
      from_date <- sprintf("%sT%sZ", lubridate::date(lubridate::as_datetime(from_date)),  hms::as_hms(lubridate::as_datetime(from_date)))
    
      second_part <- usage_last3h %>% filter(Date >= lubridate::as_datetime(now_time - 1.5*3600) & Usage == 1) %>% select(Date)
    
      min_diff_last <- which.min(abs(difftime(lubridate::as_datetime(now_time), second_part[ ,1])))
    
      to_date <- lubridate::as_datetime(second_part[min_diff_last, 1])
      to_date <- sprintf("%sT%sZ", lubridate::date(to_date),  hms::as_hms(to_date))},
      error = function(e){
        log_error(e)
      })
    
  }
  
  log_info("Checking if time interval is correct")
  
  tryCatch({
    if(find_first){
    if(nrow(first_part) == 0){
      
      from_date <- lubridate::as_datetime(now_time - 3600)
      from_date <- sprintf("%sT%sZ", lubridate::date(lubridate::as_datetime(from_date)),  hms::as_hms(lubridate::as_datetime(from_date)))
    }
    
    if(nrow(second_part) == 0){
      
      to_date <- now_time
      to_date <- sprintf("%sT%sZ", lubridate::date(to_date),  hms::as_hms(to_date))
      
    }
  
  }
    
  }, 
  error = function(e){
    log_error(e)
  })
  
  log_info("Updating module initialized")
  tryCatch({
    main_compute_daily_indicators(from_date, to_date, data_source = "influx")},
    
    error = function(e){
      log_error(e)
    })
  
  return(T)
}






