library(dplyr)
library(readr)



# GLOBAL VARIABLES
THRESHOLD <- 0.1
RECIPE_EXTRACT_PATH <- here::here("Data", "auxiliary_files", "RecipeExtract.xlsx")
OPTIMAL_SPEEDS_PATH <- Sys.getenv("optimal_speeds_fold")



load_recipe_data <- function() {
  
  recipe_extract <- readxl::read_excel(RECIPE_EXTRACT_PATH)
  
  recipe_extract[, c(1, 2, 8:19)] <- recipe_extract[, c(1, 2, 8:19)] %>%
    lapply(parse_number, locale = locale(decimal_mark = ",")) %>% 
    lapply(as.numeric)
  
  recipe_extract <- recipe_extract[, c(6, 14:18)]
  
  colnames(recipe_extract) <- c("EX_INF_recipe_ON",
                                "EX1_setpoint_speed",
                                "EX2_setpoint_speed",
                                "EX3_setpoint_speed",
                                "EX4_setpoint_speed",
                                "EX5_setpoint_speed")
  
  recipe_extract <- recipe_extract %>%
    mutate(EX1_speed_slope = NA_real_,
           EX2_speed_slope = NA_real_,
           EX3_speed_slope = NA_real_,
           EX4_speed_slope = NA_real_,
           EX5_speed_slope = NA_real_,
           Revised = TRUE)
  
  
  return(recipe_extract)
  
}



join_data_with_extract <- function(historical_selection, recipe_extract) {
  
  joint_data <-  historical_selection %>%
    full_join(recipe_extract)
  
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
      
      if ((length(unique(recipes$Revised)) == 2) && (nrow(recipes) == 2)) {
        
        idx <- ifelse(xor((recipes[recipes$Revised == TRUE, 2:6] == 0), (recipes[recipes$Revised == FALSE, 2:6] == 0)),
                      TRUE,
                      (abs(recipes[recipes$Revised == FALSE, 2:6] - recipes[recipes$Revised == TRUE, 2:6]) / recipes[recipes$Revised == TRUE, 2:6]) > THRESHOLD) 
        
        return(recipes[recipes$Revised == any(idx),])
        
      } else {
        
        return(recipes[nrow(recipes),])
        
      }
      
    }) %>% 
    bind_rows() %>% 
    select(-group_id)
  
  result <- rbind(individual_data,
                  grouped_data) %>% 
    filter(!is.na(EX_INF_recipe_ON)) %>%
    select(-Revised)
  
  return(result)
  
}



fix_slopes <- function(optimal_speeds) {
  
  for (idx in 7:11) {
    
    mask <- xor((optimal_speeds[, idx] == 0), (optimal_speeds[, idx - 5] == 0))
    
    if (sum(mask) > 0) {
      
      optimal_speeds[mask, idx] <- ifelse(optimal_speeds[mask, idx - 5] > 0,
                                          mean(unlist(optimal_speeds[, idx]), na.rm = TRUE),
                                          0)
      
    }
    
  }
  
  return(optimal_speeds)
  
}



fix_slopes_NA <- function(optimal_speeds){
  
  for (idx in 7:11) {
    
    na_mask <- is.na(optimal_speeds[, idx])
    
    # optimal_speeds[na_mask, idx] <- median(unlist(optimal_speeds[, idx]), na.rm = TRUE)
    
    if (sum(na_mask) > 0) {
      
      optimal_speeds[na_mask, idx] <- ifelse(optimal_speeds[na_mask, idx - 5] > 0,
                                             mean(unlist(optimal_speeds[, idx]), na.rm = TRUE),
                                             0)
      
    }
    
  }
  
  return(optimal_speeds)
  
}
