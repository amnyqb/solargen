#  title: "Solar Generation Forecasting using Machine Learning - Harvard Capstone Project"
#  author: "Amin Al Yaquob"
#  date: "`r Sys.Date()`"
# Function to install missing packages
install_if_missing <- function(package) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}

# List of required packages
required_packages <- c("prophet", "tensorflow", "randomForest", 
                       "caret", "lubridate", "dplyr", "ggplot2", 
                       "corrplot", "ggcorrplot", "rpart","kableExtra")

# Install missing packages
lapply(required_packages, install_if_missing)



library(tidyverse)
library(lubridate)
library(caret)
library(randomForest)


# Required libraries for downloading and unzipping files
library(tidyverse)

# Set the URL of the zip file and the destination
url <- "https://github.com/amnyqb/solar_ml/blob/main/kaggle_solardata.zip?raw=true"
destfile <- "kaggle_solardata.zip"

# Check if the file already exists, if not, download it
if (!file.exists(destfile)) {
  download.file(url, destfile, mode = "wb")
}

# Unzip the file if it hasn't been unzipped
if (!dir.exists("kaggle_solardata")) {
  unzip(destfile, exdir = "kaggle_solardata")
}

# Load the CSV files into variables
plant1_gen <- read_csv("kaggle_solardata/Plant_1_Generation_Data.csv")
plant2_gen <- read_csv("kaggle_solardata/Plant_2_Generation_Data.csv")
plant1_weather <- read_csv("kaggle_solardata/Plant_1_Weather_Sensor_Data.csv")
plant2_weather <- read_csv("kaggle_solardata/Plant_2_Weather_Sensor_Data.csv")


# Check for missing values
cat("Missing values in Plant 2 Generation Data:\n")
print(sapply(plant2_gen, function(x) sum(is.na(x))))

cat("Missing values in Plant 2 Weather Data:\n")
print(sapply(plant2_weather, function(x) sum(is.na(x))))

# Merge generation and weather data
plant2_data <- merge(plant2_gen, plant2_weather, by = "DATE_TIME")

# Preview the merged data
head(plant2_data)

# Data Cleaning and Feature Engineering

# Convert DATE_TIME to a proper datetime format and extract useful features
plant2_data <- plant2_data %>%
  mutate(DATE_TIME = as.POSIXct(DATE_TIME, format = "%Y-%m-%d %H:%M:%S"),
         hour = as.numeric(format(DATE_TIME, "%H")),
         day_of_week = as.numeric(format(DATE_TIME, "%u")),
         week_number = isoweek(DATE_TIME),
         month = as.numeric(format(DATE_TIME, "%m")),
         year = as.numeric(format(DATE_TIME, "%Y")),
         is_weekend = ifelse(day_of_week > 5, 1, 0))

# Summary after feature engineering
summary(plant2_data)


# 1. Distribution of DC Power
ggplot(plant2_data, aes(x = DC_POWER)) + 
  geom_histogram(binwidth = 100, fill = "blue", color = "grey") +
  ggtitle("Distribution of DC Power Generation") +
  xlab("DC Power (kW)") + ylab("Frequency")

# 2. DC Power vs Ambient Temperature
ggplot(plant2_data, aes(x = AMBIENT_TEMPERATURE, y = DC_POWER)) + 
  geom_point(alpha = 0.5) + 
  ggtitle("DC Power vs Ambient Temperature") + 
  xlab("Ambient Temperature (°C)") + ylab("DC Power (kW)")

# 3. DC Power vs Irradiation
ggplot(plant2_data, aes(x = IRRADIATION, y = DC_POWER)) + 
  geom_point(alpha = 0.5) + 
  ggtitle("DC Power vs Irradiation") + 
  xlab("Irradiation") + ylab("DC Power (kW)")

# 4. Correlation heatmap
cor_matrix <- cor(plant2_data %>% select(DC_POWER, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION))
ggcorrplot(cor_matrix, method = "circle", lab = TRUE, title = "Correlation Heatmap")

# 5. DC Power by Hour of Day
ggplot(plant2_data, aes(x = hour, y = DC_POWER)) + 
  geom_boxplot() + 
  ggtitle("DC Power Distribution by Hour of Day") + 
  xlab("Hour of Day") + ylab("DC Power (kW)")


# Random Forest Model
# Step 1: Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(plant2_data$DC_POWER, p = 0.8, list = FALSE)
train_data <- plant2_data[train_index, ]
test_data <- plant2_data[-train_index, ]

# Step 2: Train Random Forest model
rf_model <- randomForest(DC_POWER ~ AMBIENT_TEMPERATURE + MODULE_TEMPERATURE + IRRADIATION + hour + day_of_week, 
                         data = train_data, 
                         ntree = 100)

# Step 3: Predict on the test set
rf_pred <- predict(rf_model, test_data)

# Step 4: Calculate RMSE for Random Forest
rf_rmse <- RMSE(rf_pred, test_data$DC_POWER)
cat("Random Forest RMSE: ", rf_rmse, "\n")

# Output variable importance
importance(rf_model)


# XGBoost Model

# Load required package for XGBoost
library(xgboost)

# Step 1: Prepare data for XGBoost (convert to matrix format)
train_matrix <- as.matrix(train_data[, c("AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "hour", "day_of_week")])
test_matrix <- as.matrix(test_data[, c("AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "hour", "day_of_week")])
train_label <- train_data$DC_POWER
test_label <- test_data$DC_POWER

# Step 2: Train XGBoost model
xgb_model <- xgboost(data = train_matrix, label = train_label, max_depth = 6, eta = 0.3, nrounds = 100, objective = "reg:squarederror", verbose = 0)

# Step 3: Predict on the test set
xgb_pred <- predict(xgb_model, test_matrix)

# Step 4: Calculate RMSE for XGBoost
xgb_rmse <- RMSE(xgb_pred, test_label)
cat("XGBoost RMSE: ", xgb_rmse, "\n")

# Output feature importance
xgb.importance(model = xgb_model)



# Hyperparameter Tuning
# Random Forest hyperparameter tuning
tuned_rf <- randomForest(DC_POWER ~ AMBIENT_TEMPERATURE + MODULE_TEMPERATURE + IRRADIATION + hour + day_of_week, 
                         data = train_data, 
                         ntree = 500)

# Predict and calculate RMSE
rf_tuned_pred <- predict(tuned_rf, test_data)
rf_tuned_rmse <- RMSE(rf_tuned_pred, test_data$DC_POWER)
cat("Tuned Random Forest RMSE: ", rf_tuned_rmse, "\n")

# XGBoost hyperparameter tuning
params <- list(max_depth = 6, eta = 0.1, objective = "reg:squarederror")
xgb_tuned_model <- xgboost(data = train_matrix, label = train_label, params = params, nrounds = 200, verbose = 0)

# Predict and calculate RMSE
xgb_tuned_pred <- predict(xgb_tuned_model, test_matrix)
xgb_tuned_rmse <- RMSE(xgb_tuned_pred, test_label)
cat("Tuned XGBoost RMSE: ", xgb_tuned_rmse, "\n")


# Model Evaluation and Results
# Plot Predicted vs Actual values for Random Forest
library(knitr)  # For kable function to display tables
library(kableExtra)  # Optional for fancier tables

# Plot Predicted vs Actual values for Random Forest
ggplot(test_data, aes(x = rf_pred, y = DC_POWER)) + 
  geom_point(alpha = 0.5, color = "darkgreen") + 
  geom_abline(color = "red") + 
  ggtitle("Random Forest: Predicted vs Actual DC Power") + 
  xlab("Predicted DC Power") + 
  ylab("Actual DC Power")

# Plot Predicted vs Actual values for XGBoost
ggplot(test_data, aes(x = xgb_pred, y = DC_POWER)) + 
  geom_point(alpha = 0.5, color = "darkblue") + 
  geom_abline(color = "blue") + 
  ggtitle("XGBoost: Predicted vs Actual DC Power") + 
  xlab("Predicted DC Power") + 
  ylab("Actual DC Power")

# Create a results table
results <- data.frame(
  Model = c("Random Forest", "XGBoost", "Tuned Random Forest", "Tuned XGBoost"),
  RMSE = c(rf_rmse, xgb_rmse, rf_tuned_rmse, xgb_tuned_rmse)
)

# Display the table using knitr::kable()
kable(results, col.names = c("Model", "Root Mean Square Error (RMSE)"), 
      caption = "Model Performance Comparison") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F)

# Load required libraries
library(tidyverse)
library(xgboost)
library(lubridate)
library(gridExtra)

# Step 1: Load and process data
# Assuming 'plant2_data' is already loaded in your environment
# Ensure DATE_TIME is in proper format
plant2_data <- plant2_data %>%
  mutate(DATE_TIME = as.POSIXct(DATE_TIME, format = "%Y-%m-%d %H:%M:%S"))

# Step 2: Prepare train and test data
set.seed(123)
train_index <- createDataPartition(plant2_data$DC_POWER, p = 0.8, list = FALSE)
train_data <- plant2_data[train_index, ]
test_data <- plant2_data[-train_index, ]

# Step 3: Prepare data for XGBoost (convert to matrix format)
train_matrix <- as.matrix(train_data[, c("AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "hour", "day_of_week")])
test_matrix <- as.matrix(test_data[, c("AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "hour", "day_of_week")])
train_label <- train_data$DC_POWER
test_label <- test_data$DC_POWER

# Step 4: Train XGBoost model
xgb_model <- xgboost(data = train_matrix, label = train_label, max_depth = 6, eta = 0.3, nrounds = 100, objective = "reg:squarederror", verbose = 0)

# Step 5: Predict on the test set
xgb_pred <- predict(xgb_model, test_matrix)

# Step 6: Add predictions to test_data
test_data$xgb_pred <- xgb_pred

# Step 7: Select 12 representative days
set.seed(123)
representative_days <- sample(unique(as.Date(test_data$DATE_TIME)), 12)

# Step 8: Filter daytime hours (assuming daytime is between 6 AM to 6 PM)
test_data_daytime <- test_data %>%
  filter(hour(DATE_TIME) >= 6 & hour(DATE_TIME) <= 18)

# Step 9: Create 12 individual plots for each representative day
plots <- list()

for (day in representative_days) {
  day_data <- test_data_daytime %>% filter(as.Date(DATE_TIME) == day)
  
  p <- ggplot(day_data, aes(x = DATE_TIME)) +
    geom_line(aes(y = DC_POWER, color = "Actual"), size = 0.5) +
    geom_line(aes(y = xgb_pred, color = "Predicted"), linetype = "dashed", size = 0.5) +
    labs(title = paste("Forecast vs Actual for", day), x = "Time", y = "DC Power") +
    scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
    theme_minimal(base_size = 8) +
    theme(legend.position = "none", plot.title = element_text(size = 10))
  
  plots[[length(plots) + 1]] <- p
}

# Step 10: Arrange all 12 plots in a 3x4 grid
grid.arrange(grobs = plots, ncol = 3, top = "XGBoost Forecast vs Actual for 12 Representative Days (Daytime Hours Only)")





  
  
  