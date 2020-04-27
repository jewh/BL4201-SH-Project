
# File for simple plotting 


library(ggplot2)
library(ggpubr)

setwd("~/GitHub/BL4201-SH-Project/TestDirectory/R")

data <- read.csv("q3_states_RISdata.csv")

# will need to subset the data per data set type 

extinction <- subset(data, data$id == "Extinction")
no_extinction <- subset(data, data$id == "No Extinction")
random_noise <- subset(data, data$id == "Random Noise")

heather <- subset(data, data$id == "heather")
grass <- subset(data, data$id == "grass")

# specify which data set you want to work with 

set <- heather
head(set)

p1 <- ggplot(heather, aes(x=node, y=count, fill=factor(state))) + 
  geom_col() +
  xlab("Species") +
  ylab("Count") +
  labs(fill = "Discretisation State")
  # rremove("x.text")
p + ggtitle("Distribution of q3 States per Variable \nNo Extinction non-perturbed")

p2 <- ggplot(grass, aes(x=node, y=count, fill=factor(state))) + 
  geom_col() +
  xlab("Species") +
  ylab("Count") +
  labs(fill = "Discretisation State")


fig <- ggarrange(p1, p2, labels = c("A", "B"),
          common.legend = TRUE,
          ncol=1, nrow =2,
          legend="bottom")

annotate_figure(fig, 
                top = text_grob("Distribution of Discrete q3 States in Rothamsted Data", face="bold", size=14))

