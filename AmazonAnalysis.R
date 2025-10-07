# AmazonEmployeeAccess

library(tidyverse)

library(tidymodels)

library(vroom)

library(patchwork)

library(ggplot2)

library(recipes)

train_data <- vroom("C:\\Users\\mitch\\OneDrive\\Documents\\GitHub\\AmazonEmployeeAccess\\train.csv")

test_data <- vroom("C:\\Users\\mitch\\OneDrive\\Documents\\GitHub\\AmazonEmployeeAccess\\test.csv")


# Exploratory Plots

dplyr::glimpse(train_data) 

skimr::skim(train_data) 

DataExplorer::plot_correlation(train_data)

DataExplorer::plot_bar(train_data)

DataExplorer::plot_missing(train_data)

GGally::ggpairs(bike_data)

MGR_Approvals <- ggplot(data = train_data, aes(x = factor(MGR_ID), y = ACTION)) +
  geom_jitter(width = 0.2, height = 0.05, alpha = 0.5) +
  stat_summary(fun = mean, geom = "point", color = "red", size = 3) +
  labs(x = "Manager ID", y = "Approval (0 or 1)", 
       title = "Individual Approvals by Manager") +
  theme_minimal()

# Red points show the average approvals by manager

MGR_Approvals

MGR_hist <- ggplot(data = train_data, aes(x = MGR_ID)) +
  geom_histogram()

MGR_hist


train_data <- train_data %>%
  mutate(across(where(is.numeric) & !all_of("ACTION"), as.factor))

# Create recipe
my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  # Collapse rare categories (<0.1%)
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  # One-hot encode
  step_dummy(all_nominal_predictors())

# Prep and bake
rec_prep <- my_recipe %>% prep(training = train_data, retain = TRUE)
processed_data <- bake(rec_prep, new_data = NULL)

# Export processed dataset
vroom_write(x = processed_data, file = "./processed_train_data.csv", delim = ",")
