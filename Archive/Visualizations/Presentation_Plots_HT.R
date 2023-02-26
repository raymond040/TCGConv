library(ggplot2)
library(dplyr)
library(stringr)
library(purrr)
source("R_Code/Extract_ModelData_Function.R")

# Read in Data ----
folder <-  "Results/After_HT" #Models
df_results <- folder %>% 
  list.files(full.names = T) %>% 
  Extract_ModelData()

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
  scale_linetype_discrete(labels = c("CONCAT", "All-to-Nodes", "TCGConv"))+
  scale_color_discrete(labels = c("CONCAT", "All-to-Nodes", "TCGConv")) +
  theme_bw() +
  theme(legend.position = "bottom",
        text =  element_text(size = 15))

ggsave("AP_HT.pdf", device = "pdf", path = "Visualizations/",
       width = 6.5, height = 6, units = "in")

ggplot(df_plotting) + 
  geom_line(mapping = aes(x = Batch, y = F1_mean, linetype = Model, col = Model))+
  facet_wrap(~Dataset_Label, nrow = 2, scales = "free") +
  labs(title = "F1-Score By Model",
       y = "F1-Score") +
  scale_linetype_discrete(labels = c("CONCAT", "All-to-Nodes", "TCGConv"))+
  scale_color_discrete(labels = c("CONCAT", "All-to-Nodes", "TCGConv"))+
  theme_bw() +
  theme(legend.position = "bottom",
        text =  element_text(size = 15))
ggsave("F1_HT.pdf", device = "pdf", path = "Visualizations/",
       width = 6.5, height = 6, units = "in")



