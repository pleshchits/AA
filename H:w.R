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
###$$$$for error#########
use_condaenv('r~tensorflow')

# Добавляем для нейронной сети оптимизатор, функцию потерь, какие метрики выводить на экран (в примере выводится только точность)
SM %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy'))

SM %>% fit(train_price, train_x, epochs = 5, batch_size = 125)




##Домашка
##########################33LSTM
# Активируем пакеты
library(keras)
library(BatchGetSymbols)
library(plotly)

# Грузим котировки за 5 лет (1542 штук)
tickers <- c('%5ETA125.TA')
IL <- BatchGetSymbols(tickers = tickers,
                      first.date = '2014-03-11',
                      last.date = '2020-05-26',
                      cache.folder = file.path(tempdir(),
                                               'BGS_Cache') ) # cache in tempdir()
y <- IL$df.tickers$price.close
mIL <- data.frame(index = IL$df.tickers$ref.date, price = IL$df.tickers$price.adjusted, vol = IL$df.tickers$volume)
mIL <- mIL[complete.cases(mIL), ]

datalags = 10
train <- mIL[seq(1000 + datalags), ]
test <- mIL[1000 + datalags + seq(200 + datalags), ]
batch.size <- 50

train_x <- array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
test_x <- array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))

train_price = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))
test_price<- array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

#RNN
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 2000, output_dim = 32) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  train_x, train_price,
  epochs = 5,
  batch_size = 128,
  validation_split = 0.2
)

#LSTM

# обучаем сеть
model <- keras_model_sequential() %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model

model %>%
  compile(loss = 'mse', optimizer = 'rmsprop')

model %>% fit(train_x, train_price, epochs = 5, batch_size = batch.size)

# предсказания )
pred_out <- model %>% predict(test_x, batch_size = batch.size) %>% .[,1]
p <- data_frame(x = pred_out)
# визуализация прогнозов модели
plot_ly(mIL, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol) %>%
  add_trace(y = c(rep(NA,1000), pred_out), x = mIL$index, name = "LSTM prediction", color = 'black')

# график отклонений
plot(y.test - pred_out, type = 'line')

plot(x = y.test, y = pred_out)








