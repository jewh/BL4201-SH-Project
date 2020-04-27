# script for making large figures 

library(ggplot2)
library(ggpubr)

setwd("~/GitHub/BL4201-SH-Project/TestDirectory/R")

# need data for the extinction/noise experiment, 
# milns policy experiment and dynamic experiment

exnoise <- read.csv("extinction_noise.csv", fileEncoding = 'UTF-8-BOM')
milns <- read.csv("extinction_discretisation.csv", fileEncoding = 'UTF-8-BOM')
dynamic <- read.csv("dynamicextinctionroughness_total.csv", fileEncoding = 'UTF-8-BOM')

head(exnoise)
head(milns)
head(dynamic)

m <- ggplot(milns, aes(x=factor(Policy), y=Roughness, fill=factor(Policy))) + 
  geom_boxplot() +
  xlab("Discretisation\nPolicy") +
  ylab(expression(rho)) +
  labs(fill = "Data Set")+
  theme(legend.position = "none")+
  scale_fill_manual(values = c("#4E84C4", "#D16103"))+
  theme(axis.text=element_text(size=12))
m

n <- ggplot(exnoise, aes(x=factor(Noise.Factor), y=Roughness, fill=factor(id))) + 
  geom_boxplot() +
  xlab(expression(sigma)) +
  ylab(expression(rho)) +
  labs(fill = "Data Set")+
  theme(legend.position = "none")+
  scale_fill_manual(values = c("#F8766D", "#00BFC4", "#00BA38"))
n

d <- ggplot(dynamic, aes(x=factor(id), y=Roughness, fill=factor(id))) + 
  geom_boxplot() +
  xlab("Dataset Type") +
  ylab(expression(rho)) +
  labs(fill = "Data Set")+
  theme(legend.position = "none")+
  scale_fill_manual(values = c("#F8766D", "#00BFC4", "#00BA38"))+
  theme(axis.text=element_text(size=12))
d


legend <- get_legend(n)
legend <- as_ggplot(legend)
legend


fig <- ggarrange(n, m, d, labels = c("A", "B", "C"), nrow=3, ncol=1, 
                 common.legend = TRUE,
                 legend = "bottom")
fig

annotate_figure(fig, 
                top = text_grob(expression(paste("Effect of Noise and Extinction on ", rho )), face="bold", size=14))
