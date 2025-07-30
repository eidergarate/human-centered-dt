library(dplyr)
library(magrittr)

input_NAs_production <- function(data_pre_scaled){
  col_names <- colnames(data_pre_scaled)
  
  for(column_i in 1:ncol(data_pre_scaled)){
    if(eval(parse(text = paste0("is.numeric(data_pre_scaled$", col_names[column_i], ")")))){
      data_pre_scaled[which(is.na(data_pre_scaled[,column_i])), column_i] <- 0
    }
  }
  
  return(data_pre_scaled)
}

input_NAs_df <- function(data_pre_scaled){
  
  col_names <- colnames(data_pre_scaled)
  for(column_i in 1:ncol(data_pre_scaled)){
    if(eval(parse(text = paste0("is.numeric(data_pre_scaled$", col_names[column_i], ")")))){
      data_pre_scaled[which(is.na(data_pre_scaled[,column_i])), column_i] <- 0
    }
  }
  return(data_pre_scaled)
}