
library(car)
library(ggpubr)
library(ggplot2)

setwd("~/GitHub/BL4201-SH-Project/TestDirectory/R")
# R script to analyse roughness statistics from Python 
# Load in the data 
rough <- read.csv("bipartiteroughness_total.csv", fileEncoding = 'UTF-8-BOM')
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
sample <- subset(rough, rough$id == "hidden nodes")
pos <- subset(rough, rough$id == "observed nodes")
neg <- subset(rough, rough$id == "Random Noise")
head(sample)

L5 <- subset(rough, rough$Links == 5)
L15 <- subset(rough, rough$Links == 15)
cyclic <- subset(rough, rough$id == "cyclic ")
# 
# cyclic_L5 <- subset(L5, L5$id == "cyclic")
# cyclic_L15 <- subset(L15, L15$id == "cyclic")

# hidden_nodes <- subset(rough, rough$id == "Hidden Layer")
# observed_nodes <- subset(rough, rough$id == "All Observed")
# half_hidden <- subset(rough, rough$id == "Half Observed")

n1 <- subset(rough, rough$Noise.Factor == 1)
n10 <- subset(rough, rough$Noise.Factor == 10)

extinction <- subset(rough, rough$id == "Extinction")
pos <- subset(rough, rough$id == "No Extinction")
neg <- subset(rough, rough$id == "Random Noise")

plot(density(sample$Roughness))
plot(density(pos$Roughness))
plot(density(neg$Roughness))

# plot(density(cyclic$Roughness))
# plot(density(acyclic$Roughness))

# plot(density(hidden_nodes$Roughness))
# plot(density(observed_nodes$Roughness))
# plot(density(half_hidden$Roughness))

plot(density(extinction$Roughness))
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

rough_no_neg <- subset(rough, rough$id != "All Observed")
head(rough_no_neg)
# Now try another levene's test
leveneTest(Roughness ~ id, data=rough)

# Return p = 0.5794, therefore equal variance 

# So we can now perform a t-test

t.test(sample$Roughness, pos$Roughness)
# Get a significant difference - now see if this is reproduced across all the instances

# Find that the data is non-normal with unequal variance 
# Thus, perform a Mann-Whitney test, to see if the samples are drawn from different distributions
# get the data subsetted appropriately
# observed_hidden <- subset(rough, rough$id != "Half Observed")
# wilcox.test(Roughness ~ id, data=observed_hidden)
# 
# observed_halfobserved <- subset(rough, rough$id != "Hidden Layer")
# wilcox.test(Roughness ~ id, data=observed_halfobserved)
# 
# hidden_halfobserved <- subset(rough, rough$id != "All Observed")
# wilcox.test(Roughness ~ id, data=hidden_halfobserved)


# Now between sample and neg, and between pos and neg 
wilcox.test(Roughness ~ id, data=rough_no_neg)
head(sample)
rough_no_pos <- subset(rough, rough$id != "Positive Control")
wilcox.test(Roughness ~ id, data=rough_no_pos)

rough_no_sample <- subset(rough, rough$id != 'Sample')
wilcox.test(Roughness ~ id, data=rough_no_sample)

wilcox.test(Roughness ~ Noise.Factor, data=sample)
# Now plot the data for dissertation
p <- ggplot(rough, aes(x=factor(id), y=Roughness, fill=factor(id))) + 
  geom_boxplot() +
  xlab("Nodes Observed") +
  ylab("P") +
  labs(fill = "Nodes Observed")
p + ggtitle("Effect of Node Silencing on Static BNI Search Space")



# do a wider non-parametric ANOVA aka Kruskal-Wallis test to see if the 
# distributions are nonequal

kruskal.test(Roughness ~ id, data=rough)
