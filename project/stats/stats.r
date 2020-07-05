library(tidyverse)
library(MASS)




cip = function (paths)
{
    agg.df <- NULL
    for (path in paths)
    {
        csv <- read.csv(path)
        if (is.null(agg.df)){
            agg.df <- csv
        }else {
            agg.df <- rbind(agg.df,csv)}

    }
    as_tibble(agg.df)
}

zscore = function (score)
{
    (score-mean(score))/sd(score)
}
## test <- as_tibble(read.csv('~/NN/dio_porco.csv'))
test <- cip(list.files(path = "~/NN/resources/esn_csv", full.names = TRUE, recursive = TRUE))

test_factor <- test %>%
    mutate_if(sapply(test,is.character),as.factor)


test_select <- test_factor %>%
    select(reservoir,spectral_radius,density,squeeze_o,post_trans,post_param,metric,metric_val)



avg_selected <- test_select %>%
    group_by(reservoir,spectral_radius,density,squeeze_o,post_trans,post_param,metric) %>%
    summarize_each(funs(mean))
## %>%
##     mutate (z.val = zscore(metric_val))
    ## summarize(
    ##     ## zmetric_val = )
    ##     metric_val = mean(metric_val))
## %>%
##     mutate (z.val = ((metric_val - mean(metric_val)))/sd(metric_val))

    ## summarise_if(is.numeric,list(~mean(.),~zscore(.)))

zscores <- avg_selected %>%
    group_by(metric) %>%
    mutate(z.val = zscore(metric_val))

zcomp <- zscores %>%
    group_by(reservoir,spectral_radius,density,squeeze_o,post_trans,post_param) %>%
    summarise(z.comp = mean(z.val))
    ## mutate(metric = "zcomp",
    ##     z.comp = mean(z.val))

only_mse <- zscores %>% filter(metric == "mse" & post_trans== "sig_prob")

only_np_cor <- zscores %>% filter(( metric == "np_cor") & post_trans== "sig_prob")
    ## group_by(reservoir,spectral_radius,density,squeeze_o,post_trans,post_param) %>%
    ## summarize(metric_val = mean(metric_val)) %>%
## mutate (z.val = zscore(metric_val))

only_np_cor_mse <- avg_selected %>% filter((metric == "mse" | metric == "np_cor") & post_trans== "sig_prob")


only_np_cor$z.val <- zscore(only_np_cor$metric_val)

only_mse$z.val <- zscore(only_mse$metric_val)

zcomp_mse_cor <- only_np_cor

zcomp_mse_cor$z.val <-  only_np_cor$z.val - only_mse$z.val


f <- ggplot(zcomp) + geom_point(aes(y=reservoir,x=z.comp))+
    facet_grid( ## reservoir  ~
                    squeeze_o )

f

g <- ggplot(zcomp_mse_cor) + geom_point(aes(x=spectral_radius,y=z.val))+
    facet_grid( reservoir  ~  density )
               ## rows=vars(metric))


g


lm  <- glm(data=zcomp,z.comp ~ reservoir*density*spectral_radius*squeeze_o)
summary(lm)
## plot(lm)
lmAic <- stepAIC(lm)
lm.aov <- aov(lm)
summary(lm.aov)


lm <- glm(data = f.id, z.zcomp ~ lf * rt * ans)
summary(lm)


## %>%
##     mutate_if(sapply(test,is.integer),as.factor)

## avg_gna <- test_factor %>%
##     group_by(reservoir,density,metric,metric_val) %>%
##     summarize_each(funs(mean,sd,zscore))
##     ## summarise_if(is.numeric,list(~mean(.),~sd(.),~zscore(.)))


## avg_test <- test_factor %>%
##     select(c(reservoir,spectral_radius,density,metric,metric_val)) %>%

##     group_by(reservoir,spectral_radius,density,metric)%>%
##     summarise_each(funs(mean,sd,zscore))


    ## summarise`( zmetric_val = (metric_val -
    ##     mean(metric_val))/sd(metric_val), metric_val.sd =
    ##     sd(metric_val), metric_val = mean(metric_val))

## test_fm <- avg_test %>%
##     filter(metric != "manhattan_distance" & metric != "teacher_loss_nd"
##     & metric != "euclidian_distance")

## gna <- avg_test %>% summarise(mean,.groups = TRUE)


## avg_test$density = as.factor(avg_test$density)

## glimpse(avg_test)

## summary(select_if(avg_test,is.numeric))
