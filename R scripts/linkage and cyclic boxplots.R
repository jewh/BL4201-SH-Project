# plots for linkage and cyclicity 

library(ggplot2)
library(ggpubr)

setwd("~/GitHub/BL4201-SH-Project/TestDirectory/R")

data <- read.csv("cyclicroughness_total.csv")
head(data)

# Now subset 

linkage <- subset(data, data$id == "cyclic ")
cyclic <- subset(data, data$Links == 15)

# Now create two plot objects 

wilcox.test(Roughness ~ id, data=cyclic)
kruskal.test(Roughness ~ id, data=cyclic)

c <- ggplot(cyclic, aes(x=factor(id), y=Roughness, fill=factor(id))) + 
  geom_boxplot() +
  xlab("Network Structure") +
  ylab(expression(rho)) +
  labs(fill = "Data Set")+
  theme(legend.position = "none")+
  scale_fill_brewer(palette="Dark2")
c

l <- ggplot(linkage, aes(x=factor(Links), y=Roughness, fill=factor(Links))) + 
  geom_boxplot() +
  xlab("Number of Network Edges") +
  ylab(expression(rho)) +
  labs(fill = "Data Set")+
  theme(legend.position = "none")+
  scale_fill_brewer(palette="Accent")

# Now arrange as one figure 

fig <- ggarrange(c, l, labels = c("A", "B"),
          common.legend = FALSE,
          ncol=2, nrow =1,
          legend = "none")

annotate_figure(fig, 
                top = text_grob(expression(paste("Effect of Network Structure on ", rho)), face="bold", size=14))
