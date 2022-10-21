library(dplyr)
library(xtable)
source("R_Code/Extract_ModelData_Function.R")


# Read in Data ----
folder <-  "Results/After_HT" #Models
df_results <- folder %>% 
  list.files(full.names = T) %>% 
  Extract_ModelData()

df_overall_results <- df_results %>%
  group_by(Dataset, Model, Type) %>%
  summarise(across(AP:F1,#Variables of interest
                   list(mean = ~mean(.x, na.rm = T),
                        sd = ~sd(.x, na.rm = T)),
                   .names = "{.col}_{.fn}")) %>%  #Name of the columns
  ungroup()
#Table Processing
df_table <- df_overall_results %>%
  select(Dataset, Model, Type, starts_with("AP"), starts_with("F1")) %>% 
  mutate(
    Dataset = case_when(
      Dataset == "CC" ~ "Credit Card",
      Dataset == "MOOC" ~ "MOOC"),
    Model = case_when(
      Model == "Baseline1" ~ "CONCAT",
      Model == "Baseline2" ~ "All-to-Nodes",
      Model == "Baseline3B" ~ "TCGConv (Ours)"
    ),
    Type = case_when(
      Type == "sub" ~ "Disjoint",
      Type == "TT" ~ "Joint"
    )
  ) %>% 
  filter(Type == "Disjoint") %>%
  select(-Type) %>% 
  mutate(across(ends_with("sd"),
                ~format(.x,scientific = T, digits = 3)),
         across(ends_with("mean"),
                ~format(.x, digits = 3))) %>% 
  rename(`Average Precision Mean` = AP_mean,
         `Average Precision Standard Deviation` = AP_sd,
         `F1 Score Mean` = F1_mean,
         `F1 Score Standard Deviation` = F1_sd)

df_table_beamer <- df_table %>%
  rename(`AP Mean` = `Average Precision Mean`,
         `AP Sd` = `Average Precision Standard Deviation`,
         `F1 Mean` = `F1 Score Mean`,
         `F1 Sd` = `F1 Score Standard Deviation`)

df_table_beamer_CC <- df_table_beamer %>% 
  filter(Dataset == "Credit Card") %>% 
  select(-Dataset) %>%
  t()

df_table_beamer_MOOC <- df_table_beamer %>% 
  filter(Dataset == "MOOC") %>% 
  select(-Dataset) %>% 
  t()

#Tables
print(xtable(df_table_beamer_MOOC))
print(xtable(df_table_beamer_CC))
