# загрузка данных Yahoo finance
install.packages('quantmod')
install.packages('Quandl')
install.packages('BatchGetSymbols')
tensorflow::install_tensorflow(version = "1.13.1")
devtools::install_github("rstudio/tensorflow")

library('quantmod')
library('Quandl')
library('ggplot2')
library('plotly')
library(BatchGetSymbols)
library('keras')
library('tensorflow')

install_tensorflow()
install_keras()

#загружаем NASDAQ, 1200 наблюдений
getSymbols(Symbols = '^IXIC', 
           from = '2015-08-18', to = '2020-05-26',
           src = 'yahoo')
head(IXIC)
autoplot(IXIC[,-5], facets = NULL)


########################
#######Keras, TF########
########################
library('keras')
library('tensorflow')
library('devtools')
library('usethis')

##########SM###########

train_price <- IXIC$IXIC.Adjusted[1:1000]
test_price <-  IXIC$IXIC.Adjusted[1001:1200]
x <-  c(IXIC$IXIC.Open, IXIC$IXIC.High, IXIC$IXIC.Low, IXIC$IXIC.Volume)
train_x <- x[1:1000]
test_x <- x[1001:1200]

train_price <- as.numeric(train_price)
test_price <-  as.numeric(test_price)
train_x <-  as.numeric(train_x)
test_x <-  as.numeric(test_x)


# строим архитектуру нейронной сети
SM <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Добавляем для нейронной сети оптимизатор, функцию потерь, какие метрики выводить на экран (в примере выводится только точность)
SM %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy'))

SM %>% fit(train_price, train_x, epochs = 5, batch_size = 125)









