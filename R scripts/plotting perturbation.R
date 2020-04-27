# Plots for dynamic extinction 

library(ggplot2)
library(ggpubr)
library(forcats)

setwd("~/GitHub/BL4201-SH-Project/TestDirectory/R")

data1 <- read.csv("q3_score_mean_perturbed_extinction_networks.csv")
head(data1)

data2 <- read.csv("q3_score_mean_non_perturbed_extinction_networks.csv")
head(data2)

# Now combine the two datasets 

data <- rbind(data1, data2)

write.csv(data, "reordered.csv")

# Now load in the P metric Data

con <- read.csv("roughness_extinction_total.csv")
ran <- read.csv("perturbedextinctionroughness_total.csv")
rough <- rbind(con, ran)
head(rough)

data <- read.csv("reordered.csv")

# Now subset 

not_noise <- subset(data, data$id != "Random Noise")
noise <- subset(data, data$id == "Random Noise")
cons <- subset(not_noise, not_noise$initial_state == "constant")
rand <- subset(not_noise, not_noise$initial_state == "random")

rough_no_neg <- subset(rough, rough$id != "Random Noise")
random <- subset(rough_no_neg, rough_no_neg$initial_state == "random")
constant <- subset(rough_no_neg, rough_no_neg$initial_state == "constant")

# Get wilcox test statistic between the two types of initial state 

wilcox.test(score ~ initial_state, data=not_noise)
wilcox.test(score ~ id, data=cons)
wilcox.test(score ~ id, data=rand)

wilcox.test(Roughness ~ id, data=random)
wilcox.test(Roughness ~ id, data=constant)

# Now create plot objects

not_noise$initial_state <- factor(not_noise$initial_state, levels=c("constant", "random"))

p <- ggplot(not_noise, aes(x=initial_state, y=score, fill=factor(id))) + 
  geom_boxplot() +
  xlab("Simulation Initial State") +
  ylab("d") +
  labs(fill = "")
)
p

n <- ggplot(noise, aes(x=factor(id), y=score, fill=factor(id))) + 
  geom_boxplot() +
  xlab("") +
  ylab("d") +
  labs(fill = "")+
  theme(legend.position = "none")
n

r <- ggplot(rough_no_neg, aes(x=factor(initial_state), y=Roughness, fill=factor(id)))+
  geom_boxplot()+
  xlab("Simulation Initial State")+
  ylab(expression(rho))+
  labs(fill = "")
r

a <- ggplot(random, aes(x=factor(id), y=Roughness, fill=factor(id)))+
  geom_boxplot()+
  xlab("")+
  ylab("P")+
  labs(fill = "")
a
# Now we arrange 

fig <- ggarrange(r,p , labels = c("A", "B"),
          common.legend = TRUE,
          ncol=2, nrow =1,
          legend = "bottom")

annotate_figure(fig, 
                top = text_grob(expression(paste("Effect of Initial Simulation State on ",rho)), face="bold", size=14))


# Also wish to test if there is a difference in P between extinction and no extinction with random initial state 

