library(ggplot2)
library(ggraph)
library(dplyr)
library(igraph)
library(wesanderson)

setwd("/home/ariane/Documents/Project3/scripts/timeseries_inversion/Rplotting")
dat <- read.csv("network_disconnected.csv")

dat$date0 <- as.Date(dat$date0)
dat$date1 <- as.Date(dat$date1)

#make sure df is sorted
dat <- dat %>% arrange(date0, date1)

#turn network df into graph and add connection type as edge attribute
graph <- graph_from_data_frame(dat[,c("date0", "date1")], directed = F, vertices = NULL)
graph <- set_edge_attr(graph, "group_id", index = E(graph), factor(dat$group_id))

#make sure nodes are sorted by date
names <- sort(names(V(graph)))
graph <- permute(graph, match(V(graph)$name, names))

# plot the graph
ggraph(graph, layout = 'linear') + 
  geom_edge_arc(aes(color = group_id))+  
  geom_node_point(size = 0.1)+
  geom_node_text(aes(label = names, angle = 90), size = 2.5, hjust = 1.1)+
  theme_graph()+
  scale_edge_color_manual(values = wes_palette("Darjeeling1", n = length(unique(dat$group_id))))+
  coord_cartesian(clip = "off")+
  geom_text(aes(x = -Inf, y = Inf, label = "A: Network"),  hjust = 0, vjust = 1, size = 4.3)+
  labs(edge_color = "Group ID", edge_linetype = "Connection type")+
  theme(plot.margin = margin(0.1,0,1.5,0, unit = "cm"),
        legend.title=element_text(size=11),
        legend.text=element_text(size=8))

ggsave("graph_example.png", dpi = 300, height = 8, width = 12, units = "cm")
