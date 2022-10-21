library(ggplot2)
library(dplyr)
library(stringr)
library(purrr)
source("R_Code/Extract_ModelData_Function.R")

# Read in Data ----
folder <-  "Results/HPC_Results" #Models
df_results <- Extract_ModelData(folder)

#Process Data ----
df_summary <- Summarise_ModelResults(df_results)


#Plotting Data ----
df_plotting <- df_summary %>% 
  filter(Type == "sub") %>% 
  mutate(Dataset_Label = factor(Dataset, labels = c("Credit Card", "MOOC")))

ggplot(df_plotting) + 
  geom_line(mapping = aes(x = Batch, y = AP_mean, col = Model, linetype = Model))+
  facet_wrap(~Dataset_Label, nrow = 2, scales = "free")+
  labs(title = "Average Precision By Model",
       y = "Average Precision") +
  scale_linetype_discrete(labels = c("Concat", "All to Nodes", "TCGcov"))+
  scale_color_discrete(labels = c("Concat", "All to Nodes", "TCGcov")) +
  theme_bw()


ggplot(df_plotting) + 
  geom_line(mapping = aes(x = Batch, y = F1_mean, linetype = Model, col = Model))+
  facet_wrap(~Dataset_Label, nrow = 2, scales = "free") +
  labs(title = "F1-Score By Model",
       y = "F1-Score") +
  scale_linetype_discrete(labels = c("Concat", "All to Nodes", "TCGcov"))+
  scale_color_discrete(labels = c("Concat", "All to Nodes", "TCGcov"))+
  theme_bw()


