library(dplyr)
library(magrittr)

add_speed_suggestions <- function(recipe_ID, pred, opt_speeds_data = Sys.getenv("optimal_speeds_fold")){
  
  if(startsWith(recipe_ID, "01")){
    
    if(pred >= 0){
      optimal_speeds <- read.csv(opt_speeds_data, stringsAsFactors = F)
      
      historical_recipes <- optimal_speeds$EX_INF_recipe_ON
      
      if(recipe_ID %in% historical_recipes){
        
        optimal_values <- optimal_speeds %>% filter(EX_INF_recipe_ON == recipe_ID) %>% select(-c(EX_INF_recipe_ON))
        
        optimal_values <- optimal_values[which(optimal_values != 0)]
        
        all_strings <- c()
        
        for(i in 1:5){
          
          if(sprintf("EX%s", i) %in% substring(colnames(optimal_values), 1, 3)){
            
            Exi_setpoint <- eval(parse(text = paste0("optimal_values %>% select", sprintf("(EX%s_setpoint_speed)", i))))
            Exi_slope <- eval(parse(text = paste0("optimal_values %>% select", sprintf("(EX%s_speed_slope)", i))))
            
            EXi_string <- sprintf("EX%s setpoint: %s rpm, slope: %s rpm/s", i, round(Exi_setpoint, 2), round(Exi_slope, 2))
            all_strings <- c(all_strings, EXi_string)
          }
          
          
        }
        
        suggestion <- paste0(all_strings, collapse = "; ")
      }
      
      else{
        
        suggestion <- sprintf("New Recipe %s, no suggestion", recipe_ID)
      }
      
    }
    
    else{
      suggestion <- "No speed suggestions"
    }
    
  }
  
  else{
    suggestion <- "Sidewall, no suggestion"
  }
  
  return(suggestion)
}
