library(ggplot2)
library(dplyr)
library(randomcoloR)
library(ggrepel)
library(scico)

setwd("/home/ariane/Documents/Project3/scripts/timeseries_inversion/Rplotting")
dat <- read.csv("network_seasonal_error.csv")

#get all dates to set limits for axes - otherwise missing date combinations may lead to missing dates on the axes
dates = sort(unique(c(dat$date0, dat$date1)))
dp1 = dates[1:length(dates)-1]
dp2 = dates[2:length(dates)]

ggplot()+
  geom_tile(data = dat, aes(date0, date1, fill = error))+
  theme_bw()+
  scale_x_discrete(limits = dp1)+
  scale_y_discrete(limits = dp2)+
  scale_fill_scico(palette = "vikO", limits = c(-15, 15))+
  labs(x = "Reference date", y = "Secondary date", fill = "Error [m]")+
  geom_text_repel(aes(x = Inf, y =-Inf), label = "B: Matrix", bg.color = "white", color = "black", vjust = 1.5, hjust = -0.1, size = 4.3)+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 7), 
        axis.text.y = element_text(size=7),
        axis.title = element_text(size=13),
        legend.title=element_text(size=11),
        legend.text=element_text(size=8),
        legend.key.size = unit(0.6, "cm"),
        legend.spacing.y = unit(0.2, "cm"),
        legend.position = c(0.98, 0.15),
        legend.justification = c(0.98, 0.15),
        legend.background = element_rect(fill = "white", color = NA, linewidth = 0.2),
        legend.margin=margin(0.1,0.1,0.1,0.1, unit='cm'),
        plot.margin = margin(0.1,0.1,0.1,0.1, unit='cm'))

ggsave("matrix_example.png", dpi = 300, height = 10, width = 10, units = "cm")
