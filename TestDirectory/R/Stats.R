
# R script to analyse roughness statistics from Python 
# Load in the data 
rough <- read.csv("roughness_in0.csv")
# Check how it looks 
head(rough)
# Desire to check that each dataset is significantly different from the other 
# using ANOVA 

# However must first validate model assumptions 
# This website (http://www.sthda.com/english/wiki/one-way-anova-test-in-r#what-is-one-way-anova-test)
# suggests to perform Levene's Test to ensure variances are equal 
# Website suggests to also check that the variables are normally distributed 




