# plotting the gene reg experimental P data 

library(ggplot2)
library(ggpubr)

setwd("~/GitHub/BL4201-SH-Project/TestDirectory/R")

data <- read.csv("bipartiteroughness_total.csv")
head(data)

dynamic <- read.csv("generegdynamicroughness_total.csv")
head(dynamic)

# Now subset 

without_half_network <- subset(data, data$id != "Half Observed")

wilcox.test(Roughness ~ id, data=dynamic)
kruskal.test(Roughness ~ id, data=dynamic)

# now create plot objects
levels(data$id)
data$id <- factor(data$id,levels=c("One Layer","Two Layers","Half Network"))
  
  
p <- ggplot(data, aes(x=id, y=Roughness, fill=factor(id)))+
  geom_boxplot()+
  xlab("") +
  ylab(expression(rho)) +
  labs(fill = "")+
  theme(legend.position = "none")
p

d <- ggplot(dynamic, aes(x=factor(id), y=Roughness, fill=factor(id)))+
  geom_boxplot()+
  xlab("Nodes Observed") +
  ylab(expression(rho)) +
  labs(fill = "")+
  theme(legend.position = "none")+
  scale_fill_manual(values = c("#F8766D", "#00BA38"))
d

fig <- ggarrange(p, d , labels = c("A", "B"),
          common.legend = FALSE,
          ncol=1, nrow =2,
          legend = "none")

annotate_figure(fig, 
                top = text_grob(expression(paste("Effect of Node Silencing on ",rho)), face="bold", size=14))
