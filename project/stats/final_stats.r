library(tidyverse)
library(MASS)

spath = "~/NN/resources/esn_csv/final1"


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

res <- cip(list.files(path = spath,
                      full.names = TRUE, recursive = TRUE))

drop_colname <- names(res) %in% c("density","noise","tempo","reg")

res <- res[!drop_colname]

res[is.na(res)]<-0

res <- res %>%
    mutate_if(sapply(res,is.character),as.factor)


avg_res <- res %>%
    group_by(reservoir,leaking_rate,spectral_radius,transformer,
             t_param,t_squeeze,squeeze_o,post_trans,post_param,post_squeeze,metric) %>%
    summarise_each(funs(mean))

zres <- avg_res %>%
    group_by(metric) %>%
    mutate(z.val = zscore(metric_val))

## NOTE: only possible if all metrics have the same gradient
## (e.g. bigger m0 means good,bigger m1 means good ..... , bigger mN
## means good)
zcomp <- zres %>%
    group_by(reservoir,leaking_rate,spectral_radius,transformer,
             t_param,t_squeeze,squeeze_o,post_trans,post_param,post_squeeze,metric) %>%
    summarise(z.comp = mean(z.val))


res_cor <- avg_res %>%
    filter(metric == "np_cor")

avg_res_cor <- res_cor %>%
    group_by(reservoir,spectral_radius) %>%
    summarise_if(is.numeric,list(~sd(.),~mean(.)))

zcomp_noot <- filter(zcomp, z.comp >=
                            quantile(zcomp$z.comp)[2])



best_res_cor <- filter(res_cor, metric_val >=
                        quantile(res_cor$metric_val)[4])
best_zcomp <- filter(zcomp, z.comp >=
                            quantile(res_cor$metric_val)[4])

best_zres <- zres %>%
    group_by(metric) %>%
    filter(z.val >=
           quantile(z.val)[4])



matrix_cor_plot <- ggplot(best_res_cor,aes(spectral_radius,metric_val,color=squeeze_o))+
    geom_violin()+
    geom_smooth()+
    facet_grid(squeeze_o ~ reservoir)

matrix_zscores_plot <- ggplot(best_zres,aes(spectral_radius,z.val,color=squeeze_o))+
    geom_violin()+
    geom_smooth()+
    facet_grid( reservoir ~ metric )

matrix_zcomp_plot <- ggplot(best_zcomp,aes(spectral_radius,z.comp,color=squeeze_o))+
    geom_violin()+
    geom_smooth()+
    facet_grid(squeeze_o ~ reservoir)

trans_cor_plot <- ggplot(best_res_cor,aes(transformer,metric_val,color=squeeze_o))+
    geom_violin()+
    geom_smooth()+
    facet_grid(t_param ~ t_squeeze)

trans_zscores_plot <- ggplot(best_zres,aes(transformer,z.val,color=squeeze_o))+
    geom_violin()+
    geom_smooth()+
    facet_grid(t_param ~ t_squeeze + metric)

trans_zcomp_plot <- ggplot(best_zcomp,aes(transformer,z.comp,color=squeeze_o))+
    geom_violin()+
    geom_smooth()+
    facet_grid(t_param ~ t_squeeze)


post_trans_cor_plot <- ggplot(best_res_cor,aes(post_trans,metric_val,color=squeeze_o))+
    geom_violin()+
    geom_smooth()+
    facet_grid(post_param ~ .)

post_trans_zscores_plot <- ggplot(best_zres,aes(post_trans,z.val,color=squeeze_o))+
    geom_violin()+
    geom_smooth()+
    facet_grid(post_param ~  metric)

post_trans_zcomp_plot <- ggplot(best_zcomp,aes(post_trans,z.comp,color=squeeze_o))+
    geom_violin()+
    geom_smooth()+
    facet_grid(t_param ~ .)

leaking_cor_plot <- ggplot(best_res_cor,aes(as.factor(leaking_rate),metric_val,color=squeeze_o))+
    geom_violin()+
    geom_smooth()+
    facet_grid(spectral_radius ~ .)

leaking_zscores_plot <- ggplot(best_zres,aes(as.factor(leaking_rate),z.val,color=squeeze_o))+
    geom_violin()+
    geom_smooth()+
    facet_grid(spectral_radius ~ metric)

leaking_zcomp_plot <- ggplot(best_zcomp,aes(as.factor(leaking_rate),z.comp,color=squeeze_o))+
    geom_violin()+
    geom_smooth()+
    facet_grid(spectral_radius ~ .)

lm  <- glm(data=best_res_cor,metric_val ~   (t_squeeze  * t_param* transformer )+ leaking_rate)

summary(lm)

## plot(lm)
lmAic <- stepAIC(lm)
lm.aov <- aov(lm)
summary(lm.aov)




leaking_cor_plot

leaking_zscores_plot

leaking_zcomp_plot

post_trans_cor_plot

post_trans_zscores_plot

post_trans_zcomp_plot

trans_cor_plot

trans_zscores_plot

trans_zcomp_plot


matrix_cor_plot

matrix_zscores_plot

matrix_zcomp_plot
