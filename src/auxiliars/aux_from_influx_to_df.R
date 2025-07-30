library(dplyr)
library(magrittr)


change_from_influxList_to_df <- function(raw_data, production = T){
  
  
  if(production){
    
    raw_data %<>% data.frame(.)
    
    date_time_col <- lubridate::as_datetime(raw_data$X_time)
    
    new_colnames <- t(raw_data[1, grepl("X_field", colnames(raw_data))])
    
    values_cols_idxs <- grepl("X_value", colnames(raw_data))
    
    corrected_raw_data <- raw_data[, values_cols_idxs]
    
    colnames(corrected_raw_data) <- new_colnames
    
    corrected_raw_data$Date <- date_time_col
  }
  
  else{
    suppressMessages(lapply(raw_data, FUN = function(var_df){
      
      date_time_col <- lubridate::as_datetime(var_df$time)
      
      new_colnames <- colnames(var_df)[grepl("EX", colnames(var_df))]
      
      values_cols_idxs <- grepl("EX", colnames(var_df))
      
      corrected_raw_data <- data.frame(var_df[, values_cols_idxs], stringsAsFactors = F)
      
      colnames(corrected_raw_data) <- new_colnames
      
      corrected_raw_data$Date <- date_time_col
      
      return(corrected_raw_data)
    }) %>% bind_cols() -> corrected_raw_data)
    
    time_vars <- colnames(corrected_raw_data)[grepl("Date", colnames(corrected_raw_data))]
    date_col <- corrected_raw_data %>% select(time_vars[1])
    corrected_raw_data %<>% select(-c(all_of(time_vars)))
    corrected_raw_data$Date <- date_col[ ,1]
    corrected_raw_data %<>% tidyr::drop_na()
  }
  
  return(corrected_raw_data)
}