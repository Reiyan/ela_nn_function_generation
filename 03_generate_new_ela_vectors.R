wd = dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wd)
library(tidyverse)
library(foreach)
ela_col_names = c('ela_distr.skewness', 'ela_meta.quad_simple.adj_r2', 'fitness_distance.fitness_std', 'ela_meta.lin_w_interact.adj_r2', 'nbc.nn_nb.sd_ratio', 'nbc.nb_fitness.cor', 'ela_meta.lin_simple.adj_r2', 'ela_meta.quad_w_interact.adj_r2')


d2_data = read_csv("data/new_functions/ela_data_d3.csv") %>%
  select(c(ela_col_names, fid))

########################################################################################################################
########################################################################################################################
lm_data = read_csv("data/new_functions/ela_data_d3.csv") %>%
  select(c(ela_col_names))

cormat = as.data.frame(cor(lm_data))
cormat$feature1 = rownames(cormat)

cormat = cormat %>%
  gather("feature2", "value", ela_distr.skewness:ela_meta.quad_w_interact.adj_r2)
ggplot(cormat, aes(feature1, feature2, fill = value, label = round(value, 2))) +
  geom_tile() +
  geom_text() +
  theme(axis.text.x = element_text(angle = 90)) +
  scale_fill_gradient(low = "white", high = "red")

# Build linear model to create new ELA feature vectors which are feasible
lm_lin_simple = lm(ela_meta.lin_simple.adj_r2 ~ ela_meta.quad_simple.adj_r2, data = lm_data)
lm_lin_inter = lm(ela_meta.lin_w_interact.adj_r2 ~ ela_meta.quad_simple.adj_r2, data = lm_data)
lm_quad_inter = lm(ela_meta.quad_w_interact.adj_r2 ~ ela_meta.quad_simple.adj_r2, data = lm_data)
lm_nbc_cor = lm(nbc.nb_fitness.cor ~ ela_meta.quad_w_interact.adj_r2, data = lm_data)
lm_nbc_sd = lm(nbc.nn_nb.sd_ratio ~ ela_meta.quad_w_interact.adj_r2, data = lm_data)

########################################################################################################################
########################################################################################################################
# Build PCA space on original BBOB data
pca = prcomp(d2_data[, -9], scale. = T, center = T)
new_data = as.data.frame(pca$x)
new_data$fid = as.factor(d2_data$fid)
exp_var = cumsum(pca$sdev^2/sum(pca$sdev^2))

########################################################################################################################
########################################################################################################################
# Generate random ELA feature vectors based on random values for three features and predict the rest via the linear
# models
set.seed(100)
n = 100000
ela_distr = runif(n = n, min = -3, max = 6)
quad_simple = runif(n, 0, 1)
fitness_std = runif(n, 0, 0.5)
lin_inter = predict(lm_lin_inter, newdata = data.frame(ela_meta.quad_simple.adj_r2 = quad_simple))
lin_simple = predict(lm_lin_simple, newdata = data.frame(ela_meta.quad_simple.adj_r2 = quad_simple))
quad_inter = predict(lm_quad_inter, newdata = data.frame(ela_meta.quad_simple.adj_r2 = quad_simple))
nbc_sd_ratio = predict(lm_nbc_sd, newdata = data.frame(ela_meta.quad_w_interact.adj_r2 = quad_inter))
nb_cor = predict(lm_nbc_cor, newdata = data.frame(ela_meta.quad_w_interact.adj_r2 = quad_inter))

possible_ela_vectors = data.frame(ela_distr.skewness = ela_distr, ela_meta.quad_simple.adj_r2 = quad_simple,
                                  fitness_distance.fitness_std = fitness_std, ela_meta.lin_w_interact.adj_r2 = lin_inter,
                                  nbc.nn_nb.sd_ratio = nbc_sd_ratio, nbc.nb_fitness.cor = nb_cor,
                                  ela_meta.lin_simple.adj_r2 = lin_simple, ela_meta.quad_w_interact.adj_r2 = quad_inter)


########################################################################################################################
########################################################################################################################
# Calculate the distances between BBOB ELA and New ELA
# Select only the 0.9999 quantile, i.e., which are the furthest away
library(fields)
distances = rdist(d2_data[, -9], possible_ela_vectors)
distances = apply(distances, 2, mean)

filtered_possible = possible_ela_vectors[which(distances >= quantile(distances, 0.9999)[1]), ]

# Sample five random vectors out of the filtered values
set.seed(2134)
idx = sample(1:length(filtered_possible), 5)

filtered_possible = filtered_possible[idx, ]
#write.csv(last_try, "filtered_ela_vectors3.csv", row.names = F)

########################################################################################################################
########################################################################################################################
# Generate PCA Plort
filtered_possible = read.csv("data/new_functions/filtered_ela_vectors.csv")
# create new dataset with all:
filtered_possible$fid = 25
filtered_possible = filtered_possible %>%
  rbind(d2_data) %>%
  mutate(fid = as.factor(fid))

pca = prcomp(select(filtered_possible, -fid), scale. = T, center = T)
new_data = as.data.frame(pca$x)
new_data$fid = as.factor(filtered_possible$fid)
exp_var = cumsum(pca$sdev^2/sum(pca$sdev^2))

label_data = new_data %>%
  filter(fid != 25) %>%
  group_by(fid) %>%
  summarize_all(mean) %>%
  ungroup()

novel_points = new_data %>%
  filter(fid == 25) %>%
  mutate(fid = "new") %>%
  mutate(n = 1:5)

new_data2 = new_data %>%
  filter(fid != 25)


ggplot() +
  geom_point(data = new_data2, mapping = aes(PC1, PC2, color = fid, label = fid), alpha = 0.5) +
  geom_point(data = novel_points, mapping = aes(PC1, PC2), color = "black", size = 3) +
  geom_text(data = novel_points, mapping = aes(PC1, PC2, label = n), nudge_y = 0.1, nudge_x = 0.4) + 
  geom_text(data = label_data, mapping = aes(PC1, PC2, label = fid), color = "black") +
  theme_light() +
  labs(x = paste("PC1 -", round((pca$sdev[1]^2)/sum(pca$sdev^2), 2), "explained variance"), 
       y = paste("PC2 -", round((pca$sdev[2]^2)/sum(pca$sdev^2), 2), "explained variance")) +
  guides(color=guide_legend(title="FID"))
