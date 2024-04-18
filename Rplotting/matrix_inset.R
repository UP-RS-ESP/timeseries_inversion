library(ggplot2)
library(dplyr)
library(ggrepel)
library(patchwork)

setwd("/home/ariane/Documents/Project3/scripts/timeseries_inversion/Rplotting")
net <- read.csv("network_example.csv")
time <- read.csv("timeseries_example.csv")
time$date <- as.Date(time$date)

#get all dates to set limits for axes - otherwise missing date combinations may lead to missing dates on the axes
dates = sort(unique(c(net$date0, net$date1)))
dp1 = dates[1:length(dates)-1]
dp2 = dates[2:length(dates)]

#plot time-series
p1 <- ggplot(data = time)+
  geom_line(aes(date, disp_inversion), lwd = 0.8, color = "#FF0000")+
  geom_point(aes(date, disp_inversion), size = 0.5)+
  theme_bw()+
  scale_x_date(expand = c(0,0), limits = c(as.Date("01.11.2018", "%d.%m.%Y"), as.Date("31.12.2023", "%d.%m.%Y")))+
  annotate("label",label = "C: Matrix inset", x = as.Date("01.11.2018", "%d.%m.%Y"), y =Inf,  vjust = 1, hjust = 0,
           label.r =unit(0,"lines"),label.padding = unit(1.5, "mm"), size = 4.3)+
  labs(x = "Date", y = "Cumulative displacement [m]")+
  theme(axis.text=element_text(size=11),
        axis.title=element_text(size=13))

#plot matrix
p2 <- ggplot()+
  geom_tile(data = net, aes(date0, date1, fill = connection_type))+
  scale_fill_manual(values = c("#F2AD00", "#5BBCD6"))+
  theme_bw()+
  labs(fill = "")+
  scale_x_discrete(limits = dp1)+
  scale_y_discrete(limits = dp2)+
  theme(axis.text.x = element_blank(),
         axis.text.y = element_blank(),
         axis.title = element_blank(),
         axis.ticks = element_blank(),
         legend.title=element_text(size=9, hjust = 0.5, vjust = 0.5),
         legend.text=element_text(size=7, hjust = 0.5, vjust = 0.5),
         legend.key.size = unit(0.6, "cm"),
         legend.spacing.y = unit(0.2, "cm"),
         legend.spacing.x = unit(0.2, 'cm'),
         legend.position = c(0.98, 0.02),
         legend.justification = c(0.98, 0.02),
         legend.background = element_rect(fill = "white", color = NA, linewidth = 0.2), 
         legend.margin=margin(t=-0.3,l=0.1,b=0.1,r=0.1, unit='cm'),
         plot.margin=grid::unit(c(0,0,-1,-1), "mm"))

#add matrix to time-series plot
p1 + inset_element(p2, left = 0.64, bottom = 0, right = 1, top = 0.45)

ggsave("matrix_inset_example.png", dpi = 300, height = 10, width = 14, units = "cm")
