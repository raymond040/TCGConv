library(dplyr)
library(ggplot2)
source("R_Code/Extract_ModelData_Function.R")

# Read in Data ----
folder <-  "Results/After_HT" #Models
df_results <- folder %>% 
  list.files(full.names = T) %>% 
  Extract_ModelData()

#Process Data ----
df_summary <- Summarise_ModelResults(df_results)

#Statistical Test ----
df_model <- df_summary %>% 
  filter(Type == "sub") %>% 
  mutate(Dataset = as.factor(Dataset),
         Model = as.factor(Model),
         Type = as.factor(Type))

#F1 Anova ----
aov_model_f1 <- aov(F1_mean ~ Model + Dataset + Model*Dataset, data = df_model)

summary(aov_model_f1 )

pairwise_results_f1 = TukeyHSD(aov_model_f1)

pairwise_results_f1$`Model:Dataset` %>% 
  as.data.frame() %>% 
  filter(`p adj` < .05)

#AP Anova 
aov_model_ap <- aov(AP_mean ~ Model + Dataset + Model*Dataset, data = df_model)

summary(aov_model_ap)

pairwise_results_ap = TukeyHSD(aov_model_ap)

pairwise_results_ap$`Model:Dataset` %>% 
  as.data.frame() %>% 
  filter(`p adj` < .05)

#multcomp::glht(aov_model_f1, linfct = mcp(Dataset = "Tukey"))

#Could probably do contrasts do not not was p value on unnecessary tests and could use mancova since
#AP and F1 are related

#Plotting ----
ggplot(data = df_model)+
  geom_histogram(aes(x = F1_mean, fill = Model))+
  facet_wrap(~Dataset)
