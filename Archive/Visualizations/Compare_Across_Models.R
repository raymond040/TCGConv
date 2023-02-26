library(ggplot2)
library(dplyr)
library(stringr)
library(purrr)
source("R_Code/Extract_ModelData_Function.R")
## Extracting CSV -------------
folder <-  "Results/HPC_Results" #Models

df_results <- Extract_ModelData(folder)

# Looking  ----
# examine <- df_results %>% 
#   filter(Model == "Baseline3B")

# df_results %>% 
#   group_by(Dataset, Model) %>% 
#   summarise(across(AP:F1,
#             list(count_na = ~sum(is.na(.x)))))
#Statistics ----

df_summary <- df_results %>% 
  group_by(Dataset, Model, Type, Batch) %>%
  summarise(across(AP:F1,#Variables of interest
                   list(mean = ~mean(.x, na.rm = T), sd = sd,
                        lower = ~mean(.x) - sd(.x),#Used in plotting
                        higher = ~mean(.x) + sd(.x)), 
                   .names = "{.col}_{.fn}")) %>%  #Name of the columns
  ungroup() %>% 
  mutate(Model_Type = str_c(Model,"_", Type))

counts <- df_results %>%
  group_by(Dataset, Model, Type, Batch) %>%
  summarise(count = n())


df_overall_results <- df_results %>%
  group_by(Dataset, Model, Type) %>%
  summarise(across(AP:F1,#Variables of interest
                   list(mean = ~mean(.x, na.rm = T)),
                        #lower = ~mean(.x) - sd(.x),#Used in plotting
                        #higher = ~mean(.x) + sd(.x)),
                   .names = "{.col}_{.fn}")) %>%  #Name of the columns
  ungroup()
count_na <- df_results %>% 
  group_by(Dataset, Model, Type) %>% 
  summarise(across(.fns = ~sum(is.na(.x))))

# Plotting ----

g = ggplot(data = df_summary, aes(x = Batch))


g + 
  geom_line(aes(y = AP_mean, col = Model)) +
  facet_wrap(~Dataset+Type,scales = "free", nrow =2) +
  labs(title = "Average Precision Across Experiments by Model and Type",
       y = "Average Precision")
#ggsave("Visualizations/Average_Precision_line_prelim.png", width = 10, height = 4)

g + 
  geom_line(aes(y = P_mean, col = Model)) +
  facet_wrap(~Dataset+Type,scales = "free", nrow =2) +
  labs(title = "Precision Across Experiments by Model and Type",
       y = "Precision")


g + 
  geom_line(aes(y = R_mean, col = Model)) +
  facet_wrap(~Dataset+Type,scales = "free", nrow =2) +
  labs(title = "Recall Across Experiments by Model and Type",
       y = "Recall")

g + 
  geom_line(aes(y = F1_mean, col = Model)) +
  facet_wrap(~Dataset+Type,scales = "free", nrow =2) +
  labs(title = "F1-Score Across Experiments by Model and Type",
       y = "F1-Score")


# Boxplots ----

ggplot(data = df_summary) +
  geom_violin(mapping = aes(y = AP_mean, x = Model)) +
  facet_wrap(~Dataset + Type, scales = "free") +
  labs(title = "Average Precision Across Experiments by Model and Type",
       y = "Average Precision")


ggplot(data = df_summary) +
  geom_violin(mapping = aes(y = P_mean, x = Model)) +
  facet_wrap(~Dataset + Type, scales = "free") +
  labs(title = "Precision Across Experiments by Model and Type",
       y = "Precision")


ggplot(data = df_summary) +
  geom_violin(mapping = aes(y = R_mean, x = Model)) +
  facet_wrap(~Dataset + Type, scales = "free") +
  labs(title = "Recall Across Experiments by Model and Type",
       y = "Recall")

  

ggplot(data = df_summary) +
  geom_violin(mapping = aes(y = F1_mean, x = Model)) +
  facet_wrap(~Dataset + Type, scales = "free")+
  labs(title = "F1-Score Across Experiments by Model and Type",
     y = "F1-Score")

