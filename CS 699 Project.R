library(tidyverse)

df = read_csv("Heart Attack Detection Dataset.csv", col_names = TRUE)
spec(df)

# Summary of the initial data set

summary(df)

# Renaming columns in the data set

df_1 = rename(df, HLTHPLN = HLTHPLN1, PERSDOC = PERSDOC2, CHECKUP = CHECKUP1, EXERANY = EXERANY2,
       SLEPTIM = SLEPTIM1, STROKE = CVDSTRK3, ASTHMA = ASTHMA3, DEPRDIS = ADDEPEV3,
       EDUCATION = EDUCAG, EMPLOYMENT_STATUS = EMPLOY1, INCOME_CAT = INCOMG, GENDER = SEX,
       AGE_CAT = AGEG5YR, BMI_CAT = BMI5CAT, SMOKER_CAT = SMOKER3, HVY_DRINK = RFDRHV7,
       HEART_ATTACK = CVDINFR4)

# Checking dimensions of the data set

dim(df_1)

# Checking null values in columns

sapply(df_1, function(x) sum(is.na(x)))

# Too many null values in pregnant column so we will remove that column.

df_1 = select(df_1, -c(PREGNANT, POORHLTH))

# Selecting rows with HEART_ATTACK = 1 & HEART_ATTACK = 2

df_2 = df_1 %>%
  filter(HEART_ATTACK %in% c(1,2))

sapply(df_2, function(x) sum(is.na(x)))

# Filtering values from data set. Imputing missing values with median value of column.

for (col_num in 1:(ncol(df_2) - 1)) {
  a = which.max(table(df_1[df_1$HEART_ATTACK %in% c(1), col_num]))
  names(a) = NULL
  df_2[is.na(df_2[,col_num]) & df_2$HEART_ATTACK %in% c(1), col_num] = a
  b = which.max(table(df_1[df_1$HEART_ATTACK %in% c(1), col_num]))
  names(b) = NULL
  df_2[is.na(df_2[,col_num]) & df_2$HEART_ATTACK %in% c(2), col_num] = b
}

# Checking Null Values in all columns

sapply(df_2, function(x) sum(is.na(x)))

# Checking dimensions of data set again

dim(df_2)

# Checking distribution of class variable

df_2 %>% count(HEART_ATTACK)

# Changing values of variables

df_2[df_2["HEART_ATTACK"] == 1, "HEART_ATTACK"] = 3
df_2[df_2["HEART_ATTACK"] == 2, "HEART_ATTACK"] = 1
df_2[df_2["HEART_ATTACK"] == 3, "HEART_ATTACK"] = 2

path = paste(getwd(), "/Full Clean Data.csv", sep = '')

write_csv(df_2, path)
