# File for plotting mutliple box plots

library(ggplot2)
library(ggpubr)

setwd("~/GitHub/BL4201-SH-Project/TestDirectory/R")

data <- read.csv("extinction_noise.csv")

# Now subset the data 

random_noise <- subset(data, data$id == "Random Noise")

no_noise <- subset(data, data$id != "Random Noise")
s1 <- subset(no_noise, Noise.Factor == 1)
s10 <- subset(no_noise, Noise.Factor == 10)

# Now create plots


noise <- ggplot(random_noise, aes(x=factor(id), y=Roughness, fill=factor(id))) + 
  geom_boxplot() +
  xlab("") +
  ylab("P") +
  labs(fill = "Data Set")+
  theme(legend.position = "none")+
  scale_fill_manual(values=c("#99BA38"))+
  rremove("legend")

p1 <- ggplot(s1, aes(x=factor(id), y=Roughness, fill=factor(id))) + 
  geom_boxplot() +
  xlab("") +
  ylab("P") +
  labs(fill = "Data Set")+
  theme(legend.position = "none")+
  rremove("legend")

p10 <- ggplot(s10, aes(x=factor(id), y=Roughness, fill=factor(id))) + 
  geom_boxplot() +
  xlab("") +
  ylab("P") +
  labs(fill = "Data Set")+
  theme(legend.position = "none")+
  rremove("legend")

p <- ggplot(no_noise, aes(x=factor(Noise.Factor), y=Roughness, fill=factor(id))) + 
  geom_boxplot() +
  xlab(expression(sigma)) +
  ylab("P") +
  labs(fill = "Data Set")+
  theme(legend.position="right")
  

# Now arrange in one figure 
# First extract the legend from p 

leg <- get_legend(p)
leg <- as_ggplot(leg)
leg

fig <- ggarrange(noise, p10, p1, p,leg, labels = c("A", "B", "C", "D"),
          common.legend = FALSE,
          ncol=2, nrow =3,
          legend = "none")+
          as_ggplot(leg)

annotate_figure(fig, 
                top = text_grob("Effect of Noise and Extinction on P", face="bold", size=14))


