library(influxdbclient)
library(dplyr)
library(magrittr)
library(logger)
source(here::here("src", "main", "main_production.R"))
source(here::here("src", "auxiliars", "influxDB_connection.R"))
source(here::here("src", "auxiliars", "aux_from_influx_to_df.R"))
httr::set_config(httr::config(ssl_verifypeer=0,ssl_verifyhost=0))

con <- connect_to_influxDB()

main_query_and_predict <- function(con){
  
  table_usage_name <- Sys.getenv("table_usage")
  
  log_info("Querying production SQL to get timings for influx query")
  con_production_SQL <- connect_to_production_usage_SQL()
  
  last_usage_query <- paste0("SELECT * FROM ", table_usage_name, " WHERE Date = (SELECT MAX(Date) FROM ", table_usage_name, ");")
  prev_obs <- RJDBC::dbGetQuery(con_production_SQL, last_usage_query)
  
  disconnect_from_production_SQL(con_production_SQL)
  
  log_info("Calculating from to dates for influx query")
  if(nrow(prev_obs) == 0){
    now_time <- lubridate::as_datetime(Sys.time() - 1)
    from_time <- now_time - 10
    
    from_date <- sprintf("%sT%sZ", lubridate::date(from_time),  hms::as_hms(from_time))
    to_date <- sprintf("%sT%sZ", lubridate::date(now_time),  hms::as_hms(now_time))
  }
  else{
    from_time <- lubridate::as_datetime(prev_obs$Date)
    from_date <- sprintf("%sT%sZ", lubridate::date(from_time), hms::as_hms(from_time))
    now_time <- lubridate::as_datetime(Sys.time() - 1) 
    to_date <- sprintf("%sT%sZ", lubridate::date(now_time),  hms::as_hms(now_time))
  }
  if(difftime(now_time, from_time, unit = "secs") > 10*60){
    from_time <- now_time - 10*60
    from_date <- sprintf("%sT%sZ", lubridate::date(from_time), hms::as_hms(lubridate::as_datetime(from_time)))
  }
  
  
  log_info("Influx querying")
  tryCatch({
    
    last_raw_data <- influxDB_query_production_from_to(con, from_date, to_date, uc_id, influx_bucket, influx_measurement)
    
    last_raw_data %<>% change_from_influxList_to_df(., F)
    

    },
    error = function(e){
     log_error("Influx query is not valid or there has been a problem") 
    }
  )
  main_production(last_raw_data = last_raw_data, prev_obs = prev_obs)
  
}