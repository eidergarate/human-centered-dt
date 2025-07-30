library(dplyr)
compute_proportions_categorical <- function(cat_data){
  
  cat_proportions <- cat_data %>% group_by(---) %>% 
    summarise(prop = n()/nrow(cat_data)) #Confidential variables
 
 return(cat_proportions)
 
}

get_n_cat_samples <- function(cat_data, n){
  
  cat_proportions <- compute_proportions_categorical(cat_data)
  
  cat_proportions %<>% filter(!(---)) #Confidential variables
  
  max_used_comb <- which.max(cat_proportions$prop)
  
  samples_allEX <- rep(max_used_comb,  n)
  
  samples_data <- data.frame(matrix(0, nrow = n, ncol = 4))
  colnames(samples_data) <- colnames(cat_proportions)[1:4]
  samples_data['id'] <- samples_allEX
  
  samples_data %<>% mutate(---) #Confidential variables
  
  return(samples_data)
}


