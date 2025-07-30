library(dplyr)
library(DBI)
library(magrittr)
library(ExtrucExtract)
source(here::here("src", "auxiliars", "influxDB_connection.R"))
source(here::here("src", "auxiliars", "aux_connect_production_SQL.R"))
source(here::here("src", "auxiliars", "aux_retraining_decider.R"))
source(here::here("src", "auxiliars", "aux_update_error_table.R"))
source(here::here("src", "auxiliars", "aux_version_names.R"))
source(here::here("src", "auxiliars", "aux_from_influx_to_df.R"))
httr::set_config(httr::config(ssl_verifypeer=0,ssl_verifyhost=0))

get_daily_raw_data <- function(data_source, from_date, to_date){
  
  if(data_source == "influx"){
    con <- connect_to_influxDB()
    
    raw_data <- influxDB_query_production_from_to(con, from_date, to_date, uc_id, influx_bucket, influx_measurement)
    
    object.size(raw_data)
    
    raw_data %<>% change_from_influxList_to_df(., production = F)
    }
  return(raw_data)
}


main_compute_daily_indicators <- function(from_date, to_date, data_path_scaler, model_path, 
                                 feedback_table_name, importances_path,
                                 optimal_speeds_path, host,
                                 port, user, 
                                 password,  historic_DB_name, 
                                 preprocessed_table_name, data_source){
  
  log_info("Initializing retraining and historical DB updating module")

  from_date <- sprintf("%sT%sZ", lubridate::date(lubridate::as_datetime(from_date)), hms::as_hms(lubridate::as_datetime(from_date)))
  to_date <- sprintf("%sT%sZ", lubridate::date(lubridate::as_datetime(to_date)), hms::as_hms(lubridate::as_datetime(to_date)))
  
  log_info("Getting last model and scaler versions")
  tryCatch({
    model_path_todate <- get_actual_model_path(model_path)},
    error = function(e){
     log_error(e)
     log_error("ERROR: Problem getting last model version")
    }
  )
  tryCatch({
    scaler_path_todate <- get_actual_scaler_version(data_path_scaler, model_path)},
    error = function(e){
     log_error(e)
     log_error("ERROR: Problem getting last scaler version")
    }
  )
  
  log_info("Querying last day's data from influx.")
  tryCatch({
    raw_data <- get_daily_raw_data(data_source, from_date, to_date)
    log_info("Number of rows and columns of raw data")
    log_info(object.size(raw_data))
    },
    error = function(e){
     log_error(e)
     log_error("ERROR: Problem when connecting to Influx and getting last day's data.")
    }
  )
    
  
  #preprocess raw data
  log_info("Correcting raw data and getting indicators")
  tryCatch({
    preprocessed_data_df_and_str <- ExtrucExtract::read_raw_and_preprocess(raw_data, quality_filter, deployment = T)},
    error = function(e){
     log_error(e)
     log_error("ERROR: Problem when preprocessing raw data collected from the influx.")
    }
  )
  
  preprocessed_data <- preprocessed_data_df_and_str$data %>% ungroup(.)
  
  log_info("Updating historical DB")
  tryCatch({
    ExtrucExtract::upload_preprocessed(preprocessed_data, preprocessed_data_df_and_str$columns_types, host, port, user, password, historic_DB_name, preprocessed_table_name)},
    error = function(e){
     log_error(e)
     log_error("ERROR: Problem uploading data to the historical SQLite DB. Check the connection.")
    }    
  )
  
  log_info("Connecting to production SQL DB to get feedback")
  tryCatch({
    con <- connect_to_production_SQL(host_prodDB, port_prodDB, db_name_prodDB, user_prodDB, password_prodDB)
    log_info("We don't have feedback yet.")
    },
    
    error = function(e){
     log_error(e)
     log_error("ERROR: Problem when connecting to Production MySQL to get feedback table")
    }
  )
  
  log_info("Updating error table in the historical DB")
  tryCatch({
    update_error_table(feedback_table_name, importances_path, scaler_path_todate, model_path_todate, preprocessed_data)},
    error = function(e){
     log_error(e)
     log_error("ERROR: Problem updating error table of the historical SQLite DB")
    }
  )
  log_info("Disconnecting from production SQL")
  tryCatch({
    },
    error = function(e){
     log_error(e)
     log_error("ERROR: Problem disconnection from production MySQL DB.")
    }
  )

  tryCatch({
    retraining_decider(model_path, data_path_scaler, importances_path, dir_historicDB, historic_DB_name, preprocessed_table_name)},
    error = function(e){
     log_error(e)
     log_error("ERROR: Problem in the retraining module.")
    }
  )
  
}




