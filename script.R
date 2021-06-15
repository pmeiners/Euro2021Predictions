library(tidyverse)
library(readxl)
library(cmdstanr)
library(tidybayes)
library(posterior)
library(viridis)
library(bayesplot)
library(pROC)

#Load in data
euro_data = read_csv('data/qualifying_round_games.csv') 

# Standardize the points data
ranking_data = read_csv('data/rankings.csv') %>% 
               mutate(prior_score = as.numeric(scale(pts))) %>% 
               arrange(team)

# extract data for model
teams = ranking_data$team
nteams = length(teams)
ngames = nrow(euro_data)
team1 = match(euro_data$team1, teams)
team2 = match(euro_data$team2, teams)
score1 = euro_data$score1
score2 = euro_data$score2
b_mean = 0 #prior mean for beta, I'm very skeptical these past rankings are very informative.
b_sd = 0.05 #prior sd for beta.
prior_score = ranking_data$prior_score

# Store data in a list to pass to Stan
model_data = list(
  nteams = nteams,
  ngames = ngames,
  team1 = team1,
  team2 = team2,
  score1 = score1,
  score2 = score2,
  prior_score = prior_score,
  b_mean = b_mean,
  b_sd = b_sd
)

# Instantiate model and run sampling.
model = cmdstan_model('models/euro_raw_dif.stan')
fit = model$sample(model_data, parallel_chains=4, seed=849210, show_messages = F)

fit$cmdstan_diagnose()
y = score1-score2

yrep = as_draws_matrix(fit$draws('yppc'))[1:100, ]

ppc_bars(y, yrep, prob=0.95)+xlim(-10, 10)

a = fit$draws('a') %>% as_draws_df
sigma_y = fit$draws('sigma_y')
est_df = fit$draws('df')

# Given two country names (e.g. Italy and Turkey),
# extract goal differential posterior draws from model
goal_diff = function(teamA, teamB, do_round=T){
  set.seed(0)
  ixa = match(teamA, teams)
  ixb = match(teamB, teams)
  ai = a[, ixa]
  aj = a[, ixb]
  random_outcome = (ai - aj) + rt(nrow(ai-ai), est_df)*sigma_y
  if(do_round){
    round(pull(random_outcome))
  }
  else{
    pull(random_outcome)
  }
  
  
}

# Estimate the probaility teamA beats teamB
prob_win = function(teamA, teamB){
  random_outcome = goal_diff(teamA, teamB)
  mean(random_outcome>0)
}

# Compute P(Team A wins), P(Team B wins), P(Draw) for two teams
predict = function(teamA, teamB){
  gd = goal_diff(teamA, teamB)
  outcome_space = tibble(outcome = c('A Wins', 'B Wins', 'Draw'),
                         result = c(1, -1, 0))
  
  gdr = case_when(gd<0~-1, gd>0~1, T~0)
  
  tibble(result = gdr) %>% 
    right_join(outcome_space) %>% 
    group_by(outcome) %>% 
    summarise(n = n()) %>% 
    mutate(n = n/sum(n)) %>% 
    spread(outcome, n)
  

}

# Compute the probability team 1 wins in our training data
yhat = map2_dbl(euro_data$team1, euro_data$team2, prob_win)
# In which matches did team 1 actually win?
y = as.integer((score1>score2))

auc(multiclass.roc(y, yhat))

# Construct groups for predicting results of the group stage
group_a_teams = tibble(team = c('Italy','Switzerland','Turkey','Wales'), group = 'A')
group_b_teams = tibble(team = c('Belgium','Denmark','Finland','Russia'), group = 'B' )
group_c_teams = tibble(team = c('Austria','Netherlands','North Macedonia', 'Ukraine'), group = 'C')
group_d_teams = tibble(team = c('Croatia', 'Czech', 'England', 'Scotland'), group = 'D' )
group_e_teams = tibble(team = c('Poland', 'Slovakia','Spain', 'Sweden'), group = 'E' )
group_f_teams = tibble(team = c('France','Germany','Hungary','Portugal'), group = 'F')

# Combine individual group dataframes into a single dataframe.
groups = bind_rows(group_a_teams, group_b_teams, group_c_teams, group_d_teams, group_e_teams, group_f_teams) %>% 
         mutate(i = seq_along(team))


plot_data = full_join(groups, groups, by='group') %>% 
  filter(i.x!=i.y) %>%
  mutate(p_x_win = map2_dbl(team.x, team.y, prob_win)) %>% 
  rename(Group = group)

group_plot = plot_data %>% 
  ggplot(aes(team.y, team.x, fill = p_x_win))+
  geom_tile(size = 1, color = 'black')+
  geom_text(aes(label = scales::percent(p_x_win, accuracy = 0.1)), color = if_else(plot_data$p_x_win<0.5, 'white','black' ), size = 6)+
  facet_wrap(~Group, scales = 'free', labeller = label_both)+
  scale_fill_continuous(type='viridis',labels = scales::percent)+
  theme(aspect.ratio = 1,
        panel.background = element_blank(),
        strip.background = element_rect(fill = 'black'),
        strip.text = element_text(color = 'white', size = 8),
        plot.title = element_text(size = 12),
        plot.subtitle = element_text(size = 8),
        panel.spacing = unit(2, "lines")
        )+
  labs(y='', 
       x = '',
       title = 'Euro 2021',
       fill = 'Gewinnchancen',
       subtitle = 'Wahrscheinlichkeit Team auf y Achse schlägt Team auf x Achse')+
  guides(fill = F)


group_plot

ggsave(plot = group_plot, filename = "predictions.png", device = "png", path = "/Users/paulmeiners/Documents/RWD/Euro2021Predictions", 
width = 14, height = 14) 

