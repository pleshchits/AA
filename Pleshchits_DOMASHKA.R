

# Подготовка пакетов
install.packages('BatchGetSymbols')
install.packages('plotly')
install.packages('keras')
install.packages('tensorflow')
install.packages('minimax')
install.packages('Metrics')

library('Metrics')
library('BatchGetSymbols')
library('plotly')
library('keras')
library('tensorflow')
library('minimax')


#######################################
##### LSTM
#######################################
# Загрузка данных
tickers <- c('^IXIC')

yts <- BatchGetSymbols(tickers = tickers,
                       first.date = '2014-08-18',
                       last.date = '2020-05-26',
                       cache.folder = file.path(tempdir(),
                                                'BGS_Cache') )

# Подготовка данных
y <-  yts$df.tickers$price.close
myts <-  data.frame(index = yts$df.tickers$ref.date, price = y, vol = yts$df.tickers$volume)
myts <-  myts[complete.cases(myts), ]
myts <-  myts[-seq(nrow(myts) - 1240), ]
myts$index <-  seq(nrow(myts))

# График
plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol)

# Стандартизация
msd.price <-  c(mean(myts$price), sd(myts$price))
msd.vol <-  c(mean(myts$vol), sd(myts$vol))
myts$price <-  (myts$price - msd.price[1])/msd.price[2]
myts$vol <-  (myts$vol - msd.vol[1])/msd.vol[2]

# Деление на тестовую и тренировочную
datalags = 20
train <-  myts[seq(1000 + datalags), ]
test <-  myts[1000 + datalags + seq(200 + datalags), ]
batch.size <- 50
summary(myts)
tail(test)

# Создание массивов
x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))

x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))
dim(train)
tail(test)

#####
##### LSTM, adam, mse
#####
model <- keras_model_sequential()  %>%
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

model %>%
  compile(loss = 'mse', optimizer = 'adam', metrics = c("mse"))
model %>% fit(x.train, y.train, epochs = 10, batch_size = batch.size)
lstm_adam_mse <- 0.1191

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]
pred_out

mse(y.test, pred_out)
#1.052168
mape(y.test, pred_out)
#0.7874505
mae(y.test, pred_out)
#1.107801


#####
##### LSTM, adam, mape
#####
model <- keras_model_sequential()  %>%
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

model %>%
  compile(loss = 'mape', optimizer = 'adam')
model %>% fit(x.train, y.train, epochs = 10, batch_size = batch.size)
lstm_adam_mape <- 98.2803

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

mse(y.test, pred_out)
#1.760138
mape(y.test, pred_out)
#0.9274259
mae(y.test, pred_out)
#1.242493


#####
##### LSTM, rmsprop, mse
#####
model <- keras_model_sequential()  %>%
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

model %>%
  compile(loss = 'mse', optimizer = 'rmsprop')
model %>% fit(x.train, y.train, epochs = 10, batch_size = batch.size)
lstm_rmsprop_mse <- 0.1534

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

mse(y.test, pred_out)
#1.237044
mape(y.test, pred_out)
#0.6407407
mae(y.test, pred_out)
#0.8935128

#####
##### LSTM, rmsprop, mape
#####
model <- keras_model_sequential()  %>%
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

model %>%
  compile(loss = 'mape', optimizer = 'rmsprop')
model %>% fit(x.train, y.train, epochs = 10, batch_size = batch.size)
lstm_rmsprop_mape <- 83.6863

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

mse(y.test, pred_out)
#1.547247
mape(y.test, pred_out)
#0.7839759
mae(y.test, pred_out)
#1.155371

#######################################
##### RNN
#######################################

# Загрузка данных
tickers <- c('^IXIC')
first.date <- Sys.Date() - 360*6
last.date <- Sys.Date()

yts <- BatchGetSymbols(tickers = tickers,
                       first.date = first.date,
                       last.date = last.date,
                       cache.folder = file.path(tempdir(),
                                                'BGS_Cache') )

# Подготовка данных
y <-  yts$df.tickers$price.close
myts <-  data.frame(index = yts$df.tickers$ref.date, price = y, vol = yts$df.tickers$volume)
myts <-  myts[complete.cases(myts), ]
myts <-  myts[-seq(nrow(myts) - 1240), ]
myts$index <-  seq(nrow(myts))

# Стандартизация минимакс
myts <- data.frame(index = rminimax(myts$index), price = rminimax(myts$price), vol= rminimax(myts$vol))
myts

# Деление на тестовую и тренировочную
datalags = 20
train <-  myts[seq(1000 + datalags), ]
test <-  myts[1000 + datalags + seq(200 + datalags), ]
batch.size <- 50

# Создание массивов
x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags))

x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags))

#####
##### RNN, adam, mse
#####
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "adam",
  loss = "mse",
)

history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)
rnn_adam_mse <- 0.0791
pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

tail(y.test)
mse(y.test, pred_out)
#0.8615522
mape(y.test, pred_out)
#1.963652
mae(y.test, pred_out)
#0.2542769


#####
##### RNN, adam, mape
#####
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 2, activation = "sigmoid")
model %>% compile(
  optimizer = "adam",
  loss = "mape",
)

history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)
rnn_adam_mape <- 98.3429

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

mse(y.test, pred_out)
#0.3362341
mape(y.test, pred_out)
#0.95448
mae(y.test, pred_out)
#0.5013803

#####
##### RNN, rmsprop, mse
#####
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
)

history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)
rnn_rmsprop_mse <- 0.0792

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

mse(y.test, pred_out)
#0.08570987
mape(y.test, pred_out)
#1.990498
mae(y.test, pred_out)
#0.2535945

#####
##### RNN, rmsprop, mape
#####
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "mape",
)

history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)

rnn_rmsprop_mape <- 98.7991

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

mse(y.test, pred_out)
#0.336982
mape(y.test, pred_out)
#0.9580456
mae(y.test, pred_out)
#0.5021214






#######################################
##### SM
#######################################
tickers <- c('^IXIC')
first.date <- Sys.Date() - 360*6
last.date <- Sys.Date()

# Загрузка данных
yts <- BatchGetSymbols(tickers = tickers,
                       first.date = first.date,
                       last.date = last.date,
                       cache.folder = file.path(tempdir(),
                                                'BGS_Cache') )

# Подготовка данных
y <-  yts$df.tickers$price.close
myts <-  data.frame(index = yts$df.tickers$ref.date, price = y, vol = yts$df.tickers$volume)
myts <-  myts[complete.cases(myts), ]
myts <-  myts[-seq(nrow(myts) - 1240), ]
myts$index <-  seq(nrow(myts))

# Стандартизация минимакс
myts <- data.frame(index = rminimax(myts$index), price = rminimax(myts$price), vol= rminimax(myts$vol))
myts

# Деление на тестовую и тренировочную
datalags = 20
train <-  myts[seq(1000 + datalags), ]
test <-  myts[1000 + datalags + seq(200 + datalags), ]
batch.size <- 50

# Создание массивов
x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags))

x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags))


#####
##### SM, adam, mse
#####
sm <- keras_model_sequential() %>%
  layer_dense(units = 1000, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

sm %>% compile(
  optimizer = 'adam',
  loss = 'mse')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 50)
sm_adam_mse <- 0.0842


#####
##### SM, adam, mape
#####
sm <- keras_model_sequential() %>%
  layer_dense(units = 1000, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

sm %>% compile(
  optimizer = 'adam',
  loss = 'mape')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 50)
sm_adam_mape <- 99.5917

#####
##### SM, rmsprop, mse
#####
sm <- keras_model_sequential() %>%
  layer_dense(units = 1000, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

sm %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 50)
sm_rmsprop_mse <- 0.0841

#####
##### SM, rmsprop, mape
#####
sm <- keras_model_sequential() %>%
  layer_dense(units = 1000, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

sm %>% compile(
  optimizer = 'rmsprop',
  loss = 'mape')
sm %>% fit(x.train, y.train, epochs = 19, batch_size = 50)
sm_rmsprop_mape <-101.330


#######################################
##### Таблица
#######################################

mse <- c(lstm_adam_mse, lstm_rmsprop_mse, rnn_adam_mse, rnn_rmsprop_mse, sm_adam_mse, sm_rmsprop_mse)
mape <- c(lstm_adam_mape, lstm_rmsprop_mape, rnn_adam_mape, rnn_rmsprop_mape, sm_adam_mape, sm_rmsprop_mape)

compare <- data.frame('NN' = c(rep('SM', 2), rep('RNN', 2), rep('lstm', 2)), 
                    'optimizer' = rep(c('rmsprop', 'adam'), 3),
                    'MSE' = mse,
                    'MAPE' = mape)
compare
