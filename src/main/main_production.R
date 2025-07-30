library(reticulate)
library(logger)
source(here::here("src", "auxiliars", "main_scale_data.R"))
source_python(here::here("src", "auxiliars", "retrain_and_predict_M1.py"), convert = T, envir = globalenv())
library(ExtrucExtract)
library(magrittr)
library(DBI)
library(dplyr)
source(here::here("src", "auxiliars", "aux_correct_raw.R"))
source(here::here("src", "auxiliars", "get_interactions.R"))
source(here::here("src", "auxiliars", "influxDB_connection.R"))
source(here::here("src", "auxiliars", "aux_add_optimal_speeds.R"))
source(here::here("src", "auxiliars", 'aux_connect_production_SQL.R'))
source(here::here("src", "auxiliars", "aux_version_names.R"))
source(here::here("src", "auxiliars", "aux_input_NAs_production.R"))
source(here::here("src", "auxiliars", "aux_speed_suggestions.R"))

preprocess_data_production <- function(raw_data, data_path_scaler, optimal_speeds_path = here::here("optimal_speeds.csv"), model_path){
  log_info("Correcting last extrusion's data")
  tryCatch({
    raw_data %<>% correct_data(.)},
    error = function(e){
      log_error(e)
      log_error("ERROR: Last extrusion's data has not the expected format. There may be a problem with the timeliness, timeuniqueness and/or with the columns frecuency. Check how is coming the data from the influx.")
    }
  )
  log_info("Getting indicators")
  tryCatch({
    data_all <- ExtrucExtract::preprocessing_production(raw_data, 0.01)},
    error = function(e){
      log_error(e)
      log_error("ERROR: Is not possible to process the data using ExtrucExtract. Check the columns.") 
    }
  )
  
  log_info("Adding M2's output")
  tryCatch({
    data_all <- add_optimal_speeds_production(data_all, optimal_speeds_path)},
    error = function(e){
      log_error(e)
      log_error("ERROR: Optimal speeds cannot be added. Check the M2's csv output.")
    }
    )
  log_info("Preprocessing data for the prediction")
  tryCatch({
    data_all %<>% get_interactions_production(.)},
    error = function(e){
      log_error(e)
      log_error("ERROR: Cannot be added interactions. Check the columns.")
    }
    
  )
  
  tryCatch({
    data_path_scaler <- get_actual_scaler_version(data_path_scaler, model_path)},
    error = function(e){
      log_error(e)
      log_error("ERROR: Problem when getting last version scaler")
    }
  )
  tryCatch({
    data_all %<>% input_NAs_production(.)},
    error = function(e){
      log_error("Problem when inputting NA values before scaling")
    }
    )
  
  tryCatch({
    data_all %<>% main_scale_data(., data_path_scaler)},
    error = function(e){
      log_error(e)
      log_error("ERROR: Error scaling data. Please check pickle object, Python code and data.")
    }
  )

  return(data_all)
}

predict_readiness <- function(data_all, model_path, importances_path, date, recipe_ID){
  
  
  tryCatch({
    model_file_path <- get_actual_model_path(model_path)},
    error = function(e){
      log_error(e)
      log_error("ERROR: Problem when getting last version model") 
    }
  )
  log_info("Loading model and predicting")
  tryCatch({
    pred <- predict_M1(data_all, model_file_path, importances_path)},
    error = function(e){
      log_error(e)
      log_error("ERROR: Error when predicting the steadiness. Please check the pickle object, Python code and data.")
    }
  )
  log_info("Adding suggestions")
  tryCatch({
    suggestion <- add_speed_suggestions(recipe_ID, pred)
    
    pred <- data.frame(Date = date, Pred = as.character(pred), Suggestions = suggestion, stringsAsFactors = F)},
    error = function(e){
      log_error(e)
      log_error("ERROR: Error when getting prediction error dataframe.")
    }
  )

  return(pred)
}

main_production <- function(last_raw_data, prev_obs, data_path_scaler, 
                            model_path, table_pred_name, 
                            table_usage_name, return_pred, 
                            importances_path, optimal_speeds_path){
  log_info("Initializing production mode")
  log_info("Connecting to production SQL")
  
  last_obs <- last_raw_data[nrow(last_raw_data), ]
  tryCatch({
    con_pred <- connect_to_production_pred_SQL()
    con_usage <- connect_to_production_usage_SQL()},
    error = function(e){
      log_error(e)
      log_error("ERROR: Error when connecting to the Production MySQL DataBase")
    }
    
  )
  log_info("Checking if the extruders are in usage")
  tryCatch({
    usage <- F
    
    if(last_obs$principal_EX_speed >= 0.5){
      usage <- T
    }
    else{
      usage <- F
    }},
    error = function(e){
      log_error(e)
      log_error("ERROR: Second Extruder's Speed is NA. Not possible to complete the MySQL tables.") 
    }
  )
  

  
  if(nrow(prev_obs) == 0){
    
    log_info("usage SQL is updated")
    usage <- ifelse(usage, "0", "1")
    update_usage_query <- paste0("INSERT INTO ", table_usage_name, " VALUES (1, '", usage, "', '", lubridate::as_datetime(last_obs$Date), "')")
    
    tryCatch({
      RJDBC::dbSendUpdate(con_usage, update_usage_query)},
      error = function(e){
        log_error(e)
        log_error("ERROR: Impossible to Update usage table in the production MySQL DB.") 
      }
    )
  }
  else{
    
    prev_obs %<>% mutate(Usage = if_else(Usage == "0", T, F))
    
    if(prev_obs$Usage != usage){
      
      log_info("usage SQL is updated")
      usage <- ifelse(usage, "0", "1")
      id <- prev_obs$id + 1
      
      update_usage_query <- paste0("INSERT INTO ", table_usage_name, " VALUES (", id, ", '", usage, "', '", lubridate::as_datetime(last_obs$Date), "')")
      
      tryCatch({
        RJDBC::dbSendUpdate(con_usage, update_usage_query)},
        error = function(e){
          log_error(e)
          log_error("ERROR: Problem when updating the usage MySQL table.")
          }
      )
      
    }
    else{
      usage <- ifelse(usage, "0", "1")
    }
    
    log_info("Checking if there is needed to predict the readiness")
    if(usage != "0"){
      log_info("Prediction is initialized.")
      
      log_info("Querying MySQL to know last observations Id value")
      
      last_pred_query <- paste0("SELECT * FROM ", table_pred_name, " WHERE Date = (SELECT MAX(Date) FROM ", table_pred_name, ");")
      prev_pred <- RJDBC::dbGetQuery(con_pred, last_pred_query)
      last_id <- prev_pred$id 
      if(length(last_id) == 0){last_id <- 0}
      
      
      
      log_info("Querying influx to get all the extruder data.")
      
      if(prev_obs$Usage){
        log_info("Previous obs in Prod. SQL was in usage, only last obs is in stoppage")
        last_usage_t <- max(last_raw_data[which(last_raw_data$EX_EX2_Speed_Screw_Actual >= 0.5), ]$Date)
        
        if(!is.na(last_usage_t)){
          log_info("Stoppage has happend just few seconds ago")
          last_raw_data %<>% filter(Date > last_usage_t)
        }
        log_info("Stoppage part selected")
      }
      
      else if(!prev_obs$Usage){
        log_info("Checking if there has been a restart from the last initialization")
        last_usage_t <- max(last_raw_data[which(last_raw_data$EX_EX2_Speed_Screw_Actual >= 0.5), ]$Date)
        
        if(!is.na(last_usage_t)){
          log_info("Stoppage has happend just few seconds ago")
          last_raw_data %<>% filter(Date > last_usage_t)
        }
        log_info("Stoppage part selected")
      }
      
      log_info("All data from last obs in Prod SQL to now is in stoppage")
      
      tryCatch({
        data_all <- preprocess_data_production(last_raw_data, data_path_scaler, optimal_speeds_path, model_path)},
        error = function(e){
          log_error(e)
          log_error("ERROR: Error when preprocessing last stoppage's data.")
        }
      )
      
      tryCatch({
        recipe_ID <- last_raw_data$EX_INF_ID_Recipe_Actual[nrow(last_raw_data)]
        pred <- predict_readiness(data_all, model_path, importances_path, date = last_obs$Date[1], recipe_ID = recipe_ID)},
        error = function(e){
          log_error(e)
          log_error("ERROR: Error when predicting the steadiness.")
        }
      )
      
      #return prediction if necessary
      if(return_pred){
        ret_sol <- pred
      }else{ret_sol <- NULL}
      
      log_info("Updating Production SQL with the readiness prediction.")
      
      update_pred_query <- paste0("INSERT INTO ", table_pred_name, " VALUES (", last_id + 1, ", '", pred$Pred, "', '", pred$Suggestion, "', '", lubridate::as_datetime(pred$Date), "')")
      
      tryCatch({
        RJDBC::dbSendUpdate(con_pred, update_pred_query)},
        error = function(e){
          log_error(e)
          log_error("ERROR: Is not possible to update prediction table of the production MySQL DB.") 
        }
      )
      
    }
    
    else(ret_sol <- T)
  }
  
  log_info("Disconnecting from production SQL.")
  tryCatch({
    disconnect_from_production_SQL(con_pred)
    disconnect_from_production_SQL(con_usage)
    },
    error = function(e){
      log_error(e)
      log_error("ERROR: Is not possible to disconnect from Production MySQL DB.")}
  )
  
  return(ret_sol)
}
