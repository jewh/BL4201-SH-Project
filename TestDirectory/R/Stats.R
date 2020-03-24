
library(car)
library(ggpubr)
library(ggplot2)

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
# wikipedia suggests that a significant Levene's test => unequal variance
# Website suggests to also check that the variables are normally distributed 


# Examine density plot for each dataset
sample <- subset(rough, rough$id == "Sample")
pos <- subset(rough, rough$id == "Positive Control")
neg <- subset(rough, rough$id == "Negative Control")
head(sample)

plot(density(sample$Roughness))
plot(density(pos$Roughness))
plot(density(neg$Roughness))

# Triple check with a statistical test
shapiro.test(sample$Roughness)
# Yields p = 0.38 so normally distributed
shapiro.test(pos$Roughness)
# Yields p = 0.59 so normally distributed 
shapiro.test(neg$Roughness)
# Yields p = 0.009745 so data NOT normally distributed 
# Yet for in1 have 0.39 so normally distributed 

# However this now means we can perform unpaired t-tests between pos and sample 
# Check again that the variances of the two are equal 

rough_no_neg <- subset(rough, rough$id != "Negative Control")
head(rough_no_neg)
# Now try another levene's test
leveneTest(Roughness ~ id, data=rough_no_neg)

# Return p = 0.5794, therefore equal variance 

# So we can now perform a t-test

t.test(sample$Roughness, pos$Roughness)
# Get a significant difference - now see if this is reproduced across all the instances