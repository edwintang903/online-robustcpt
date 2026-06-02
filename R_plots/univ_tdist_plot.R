library(ggplot2)
library(tidyr)
library(dplyr)
library(tibble)

# Function to compute proportions for each row
get_proportions <- function(x) {
  total <- length(x)
  c(
    not_detected = sum(x >= 2400) / total,
    false_alarm = sum(x >= 0 & x <= 600) / total,
    detected = sum(x > 600 & x < 2400) / total
  )
}

# Read CSV 
data_bars <- read.csv("data/Latest_format/locations_v2e10all.csv")

# ==== Data treatment ====
data_bars[data_bars == -1] <- 2401

exponent_seq <- seq(0, 20, by = 0.25)
powers_of_two <- 2.0^exponent_seq

detected_long <- data_bars %>%
  mutate(snr = seq(from = 0.09, to = 0.7, by = 0.01)) %>%  #seq(from = 0.085, to = 0.495, by = 0.01)
  pivot_longer(-snr, values_to = "stoppingT")

colnames(detected_long)[2]<-"trial"

# ==== Proportion plot ====
# Group by snr and compute proportions
props <- data1 %>%
  filter(snr<=0.08) %>%
  group_by(snr) %>%
  reframe(
    tibble::as_tibble_row(get_proportions(stoppingT))
  )

props$false_alarm[1]<-props$false_alarm[1]+props$detected[1]
props$detected[1]<-0

props_longer<- props %>%
  pivot_longer(
    cols = c(not_detected, false_alarm, detected),
    names_to = "category",
    values_to = "proportion") %>%
  mutate(category = factor(category,
                           levels = c("not_detected","detected","false_alarm")))

snr_step <- min(diff(sort(unique(props_longer$snr))))

ggplot(props_longer,
       aes(x = snr,
           y = proportion,
           fill = category)) +
  geom_col(width = snr_step * 0.98,
           colour = "white",
           linewidth = 0.2) +
  geom_hline(yintercept = 0.2,
             linetype = "dashed",
             linewidth = 0.4) +
  scale_x_continuous(
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    x = expression(kappa/sigma),
    y = "Empirical probability"
  ) +
  scale_fill_manual(
    values = c(
      not_detected = "#999999",
      detected     = "#0072B2",
      false_alarm  = "#D55E00"
    )
  ) +
  theme_bw(base_size = 14) +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

# ====== Mean Detection Delay plot =====
detected_long <- data1 %>%
  filter(snr >=0.081)%>%
  filter(snr <=0.7)%>%
  filter(stoppingT > 600)

result <- detected_long %>%
  group_by(snr) %>%
  summarise(meanT = mean(stoppingT), sdT=sd(stoppingT))

result$meanT<-result$meanT-600

mod1<-lm(meanT~I(snr^(-2))+0, data=result[result$snr<=0.23,], weights = 1/sdT)
mod2<-lm(meanT~I((log(snr*4.582576*4))^(-1))+0, data=result[result$snr>=0.4,], weights = 1/sdT)

# dense grid for smooth curves
# grid0 <- data.frame(
#   snr = seq(min(result$snr),
#             max(result$snr),
#             length.out = 400))
grid <- data.frame(snr = seq(0.085,0.23,length.out = 1000))
grid2 <- data.frame(snr = seq(0.3,0.7,length.out = 1000))

grid$fit_inv_sq  <- predict(mod1,  newdata = grid)
grid2$fit_log_inv <- predict(mod2, newdata = grid2)

ggplot(result, aes(x = snr, y = meanT)) +
  geom_point(size = 1.6, colour = "black") +
  geom_line(data = grid,
            aes(y = fit_inv_sq,
                colour = "Regime 2"),
            linewidth = 0.9) +
  geom_line(data = grid2,
            aes(y = fit_log_inv,
                colour = "Regime 3"),
            linewidth = 0.9) +
  scale_x_continuous(
    breaks = seq(0.1, 0.7, by = 0.1))+
  labs(
    x = expression(kappa/sigma),
    y = "Mean detection delay",
    colour = NULL
  ) +
  scale_colour_manual(
    breaks = c("Regime 2", "Regime 3"),
    values = c(
      "Regime 2" = "pink",
      "Regime 3" = "#2C7BB6"
    )
  ) +
  theme_bw(base_size = 14) +
  theme(
    panel.grid.major = element_line(colour = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.key.width = unit(1.5, "cm"),
    legend.spacing.x = unit(0.6, "cm")
  )

# R3-R4 plot
# log-log
long_data <- data1 %>%
  filter(snr >=0.3)%>%
  filter(snr <=1e5)%>%
  filter(stoppingT > 600)

result <- long_data %>%
  group_by(snr) %>%
  summarise(meanT = mean(stoppingT), sdT=sd(stoppingT))

result$meanT<-result$meanT-600

mod1<-lm(meanT~I((log(snr*4.582576*4))^(-1))+0, data=result[result$snr<=0.7,], weights = 1/sdT)
mod2<-lm(meanT~1, data=result[result$snr>=2,])

# prediction grid
grid <- data.frame(snr = seq(0.27,0.7,length.out = 10000))
grid2 <- data.frame(snr = seq(2,100000,length.out = 1000))

grid$fit_inv <- predict(mod1, newdata = grid)
grid2$fit_inv_sq <- predict(mod2, newdata = grid2)

ggplot(result, aes(x = snr, y = meanT)) +
  theme_bw(base_size = 14) +
  annotation_logticks(sides = "b")+
  geom_point(alpha = 0.7, size = 1.8) +
  geom_line(data = grid, aes(y = fit_inv, color = "Regime 3"), #Power-law (slope = -2.63)
            linewidth = 1, linetype = "solid") +
  geom_line(data = grid2, aes(y = fit_inv_sq, color = "Regime 4"), #Inverse (slope = -1)
            linewidth = 1, linetype = "solid") +
  scale_x_log10(breaks=10^seq(0,5)) +
  labs(
    x = expression(kappa/sigma),
    y = "Mean detection delay",
    color = NULL
  ) +
  scale_color_manual(
    values = c(
      "Regime 3" = "#2C7BB6",
      "Regime 4" = "#D55E00"
    )
  ) +
  theme(
    panel.grid.major = element_line(colour = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.key.width = unit(1.5, "cm"),
    legend.spacing.x = unit(0.6, "cm")
  )
