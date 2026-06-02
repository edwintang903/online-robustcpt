library(MASS)
library(ggplot2)
library(dplyr)
library(tidyr)

cpt=500
n=2000
get_proportions <- function(x) {
  total <- length(x)
  c(
    not_detected = sum(x >= n+1) / total,
    false_alarm = sum(x >= 0 & x <= cpt) / total,
    detected = sum(x > cpt & x <= n) / total
  )
}

# Load data files
rawdata<-read.csv("data/Latest_format/hdcpt_p10_full.csv")    #Supply relevant data file

# ==== Detectability plots ====
data1 <- fulldata %>%
  pivot_longer(-kappa, values_to = "stoppingT")

props <- data1 %>%
  filter(kappa %in% round(seq(from=0,to=0.45,by=0.025),3)) %>%
  group_by(kappa) %>%
  reframe(
    tibble::as_tibble_row(get_proportions(stoppingT))
  )
props$false_alarm[1]<-props$false_alarm[1]+props$detected[1]
props$detected[1]<-0
props

props_longer<- props %>%
  pivot_longer(
    cols = c(not_detected, false_alarm, detected),
    names_to = "category",
    values_to = "proportion") %>%
  mutate(category = factor(category,
                           levels = c("not_detected","detected","false_alarm")))

snr_step <- min(diff(sort(unique(props_longer$kappa))))
ggplot(props_longer,
       aes(x = kappa,
           y = proportion,
           fill = category)) +
  geom_col(width = snr_step * 0.98,
           colour = "white",
           linewidth = 0.2) +
  geom_hline(yintercept = 0.1,
             linetype = "dashed",
             linewidth = 0.5) +
  scale_x_continuous(
    breaks = seq(0, 1.2, by = 0.1),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    x = expression(kappa),
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


# ==== Detection delay plots ====
detected_long <- data1 %>%
  filter(kappa >=0.45)%>%
  filter(stoppingT > cpt)

result <- detected_long %>%
  group_by(kappa) %>%
  summarise(meanT = mean(stoppingT), sdT=sd(stoppingT))

result$meanT<-result$meanT-cpt

r3<-(result$kappa<=0.65) #&(result$kappa<=2)
mod<-lm(log(meanT)~log(kappa), data=result[r3,])
summary(mod)

mod1<-lm(meanT~0+I(kappa^(-2.7)), weights=1/sdT^2, data=result[result$kappa<=0.65,])
#mod2<-lm(meanT~0+I(kappa), weights=1/sdT^2, data=result[r3,])
mod3<-lm(meanT~1, weights=1/sdT^2, data=result[result$kappa>=1.25,])

# prediction grid
grid <- data.frame(kappa = seq(0.45,0.65,length.out = 100))
grid$fit <- predict(mod1, newdata = grid)
#grid2 <- data.frame(kappa = seq(0.5,2,length.out = 100))
#grid2$fit <- predict(mod2, newdata = grid2)
grid3 <- data.frame(kappa = seq(1.25,2,length.out = 100))
grid3$fit <- predict(mod3, newdata = grid3)

ggplot(result, aes(x = kappa, y = meanT)) +
  geom_point(size = 1.6, colour = "black") +
  geom_line(data = grid,
            aes(y = fit,
                colour = "Regime 2"),
            linewidth = 0.9) +
  # geom_line(data = grid2,
  #           aes(y = fit,
  #               colour = "Regime 3"),
  #           linewidth = 0.9) +
  geom_line(data = grid3,
            aes(y = fit,
                colour = "Regime 4"),
            linewidth = 0.9) +
  scale_x_continuous(
    breaks = seq(0, 5, by = 0.2))+
  scale_y_continuous(
    limits=c(0,3000),
    breaks = seq(0, 3000, by = 500))+
  labs(
    x = expression(kappa),
    y = "Mean detection delay",
    colour = NULL
  ) +
  scale_colour_manual(
    breaks = c("Regime 2","Regime 4"),
    values = c(
      "Regime 2" = "pink",
      #"Regime 3" = "#2C7BB6",
      "Regime 4" = "orange")
  ) +
  theme_bw(base_size = 14) +
  theme(
    panel.grid.major = element_line(colour = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.key.width = unit(1.5, "cm"),
    legend.spacing.x = unit(0.6, "cm")
  )
