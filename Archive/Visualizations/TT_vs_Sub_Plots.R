library(ggplot2)
library(dplyr)
library(stringr)
library(purrr)

## Extracting CSV -------------
folder <-  "Results/HPC_Results/Baseline1" #Baseline 1

files <-  folder %>% 
  list.files(full = T)

sub_files <- files %>% 
  str_subset(., "MOOC")#CC for credit card

name_files <- sub_files %>% 
  str_extract(pattern = "sub|TT") %>% #if the file is sub or TT
  str_c("_",seq(length(.))) #Creates unique name for each file

sub_files <- sub_files %>% 
  set_names(name_files) #Used to let map have an ID for each file


df_results <- map_dfr(sub_files, read.csv, .id = "Experiment")

# Data processing -----
df_results <-  df_results %>% 
  mutate(Type = str_extract(Experiment,"TT|sub")) %>% 
  group_by(Experiment) %>% 
  mutate(Batch = seq(n())) %>%  #Give each row in an experiment equal to a batch value
  ungroup()

# Statistics ----

df_summary <- df_results %>% 
  group_by(Type, Batch) %>%
  summarise(across(AP:F1,#Variables of interest
                   list(mean = mean, sd = sd,
                        lower = ~mean(.x) - sd(.x),
                        higher = ~mean(.x) + sd(.x)),
                   .names = "{.col}_{.fn}")) %>%  #Name of the columns
  ungroup()

#Plotting ----
g <- ggplot(data = df_summary, mapping = aes(x = Batch, col = Type))

g + 
  geom_line(mapping = aes(y = AP_mean)) +
  geom_ribbon(aes(ymin = AP_lower, ymax = AP_higher, fill = Type),
              alpha = .25,
              col = NA) +
  labs(title = "Average Precision Concat Model across Experiments",
       y = "Average Precision")+
  scale_colour_discrete(labels = c('Sub', 'Test'))+
  scale_fill_discrete(labels = c('Sub', 'Test'))


g + geom_line(mapping = aes(y = P_mean)) +
  geom_ribbon(aes(ymin = P_lower, ymax = P_higher, fill = Type),
              alpha = .25,
              col = NA) +
  labs(title = "Precision Concat Model across Experiments",
       y = "Precision") +
  scale_colour_discrete(labels = c('Sub', 'Test'))+
  scale_fill_discrete(labels = c('Sub', 'Test'))
  

g + geom_line(mapping = aes(y = R_mean)) +
  geom_ribbon(aes(ymin = R_lower, ymax = R_higher, fill = Type),
              alpha = .25,
              col = NA) +
  labs(title = "Recall Concat Model across Experiments",
       y = "Recall") +
  scale_colour_discrete(labels = c('Sub', 'Test'))+
  scale_fill_discrete(labels = c('Sub', 'Test'))


g + geom_line(mapping = aes(y = F1_mean)) + 
  geom_ribbon(aes(ymin = F1_lower, ymax = F1_higher, fill = Type),
              alpha = .25,
              col = NA) +
  labs(title = "F1 Concat Model across Experiments",
       y = "F1 - Score") +
  scale_colour_discrete(labels = c('Sub', 'Test'))+
  scale_fill_discrete(labels = c('Sub', 'Test'))
  




