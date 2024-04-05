library(ggplot2)
library(ggraph)
library(dplyr)

setwd("/home/ariane/Documents/Project3/scripts/timeseries_inversion/Rplotting")
dat <- read.csv("network_example.csv")

dat$date0 <- as.Date(dat$date0)
dat$date1 <- as.Date(dat$date1)

#make sure df is sorted
dat <- dat %>% arrange(date0, date1)

#turn network df into graph and add connection type as edge attribute
graph <- graph_from_data_frame(dat[,c("date0", "date1")], directed = F, vertices = NULL)
graph <- set_edge_attr(graph, "connection_type", index = E(graph), dat$connection_type)

#make sure nodes are sorted by date
names <- sort(names(V(graph)))
graph <- permute(graph, match(V(graph)$name, names))

# plot the graph
ggraph(graph, layout = 'linear') + 
  geom_edge_arc(aes(color = connection_type, lty = connection_type))+  
  geom_node_point(size = 0.1)+
  geom_node_text(aes(label = names, angle = 90), size = 2.5, hjust = 1.1)+
  theme_graph()+
  scale_edge_linetype_manual(values = c("solid", "longdash"), guide = "none")+
  scale_edge_color_manual(values = c("gray50", "#FF0000"))+
  coord_cartesian(clip = "off")+
  geom_text(aes(x = -Inf, y = Inf, label = "A: Network"),  hjust = 0, vjust = 1, size = 4.3)+
  labs(edge_color = "Connection type", edge_linetype = "Connection type")+
  theme(plot.margin = margin(0.1,0,1.5,0, unit = "cm"),
        legend.title=element_text(size=11),
        legend.text=element_text(size=8))

ggsave("graph_example.png", dpi = 300, height = 8, width = 12, units = "cm")
