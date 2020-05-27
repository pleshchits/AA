# �������� ������ Yahoo finance
install.packages('quantmod')
install.packages('Quandl')
install.packages('BatchGetSymbols')
tensorflow::install_tensorflow(version = "nightly")
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")
install.packages('devtools')
install.packages('ggplot2')
install.packages('tensorflow')
install.packages('keras')
install.packages('psych')
install.packages("magrittr") 
install.packages("pak")    
install.packages('plotly')
library(magrittr) 
library(dplyr)
use_condaenv("r-tensorflow")

pak::pkg_install("rstudio/tensorflow")
library('quantmod')
library('Quandl')
library('ggplot2')
library('plotly')
library(BatchGetSymbols)
library('keras')
library('tensorflow')
library('psych')

install_tensorflow()
install_keras()

#��������� NASDAQ, 1200 ����������
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


# ������ ����������� ��������� ����
sm <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# ��������� ��� ��������� ���� �����������, ������� ������, ����� ������� �������� �� �����
sm %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy'))
sm %>% fit(train_price, train_x, epochs = 5, batch_size = 125)


########################
##########RNN###########
########################

#�������� ��
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
  
  #�������� ����� � ����������� ���������
  history <- network %>% fit(train_images, train_labels,
                             epochs = 5, batch_size = 128,
                             validation_split = 0.2)


########################
##########LSTM##########
########################
  tickers <- c('^IXIC')
NAS <- BatchGetSymbols(tickers = tickers,
                        first.date = '2015-08-18',
                        last.date = '2020-05-26',
                        cache.folder = file.path(tempdir(),
                                                 'BGS_Cache') )

y <- NAS$df.tickers$price.close
mA <- data.frame(index = NAS$df.tickers$ref.date, price = NAS$df.tickers$price.adjusted, vol = NAS$df.tickers$volume)
mA <- mA[complete.cases(mA), ]
  
datalags = 10
train <- mA[seq(1000 + datalags), ]
test <- mA[1000 + datalags + seq(200 + datalags), ]
batch.size <- 50

# ����������� ������� ��� lstm
# �� X ����������� (����� ���������� �������������/�������� �������, ���, ����� ����������)
# �� Y ����������� (����� ����������, ����� ���������� �� ������)
  
train_x <- array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
test_x <- array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
  
train_price = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))
test_price<- array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

  
# ������� ����
model_LSTM <- keras_model_sequential() %>%
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
  
model_LSTM
  
model_LSTM %>%
    compile(loss = 'mse', optimizer = 'rmsprop')

model_LSTM %>% fit(train_x, train_price, epochs = 5, batch_size = batch.size)


#######################################
#����� ������ ���� ���� ��������� ��, �������� �� ������� �/� ������������ � ����������� ����������,
#��, ��� ��� ��������� ��������� ������, ��������� � ���������� �������, � ��� ����������� ���� ������ ������,
#��������� �������������� ������, ��� ������� ������� ����� ���� �������� �����,
#���������� �������� �������� �� ��������. ������ ���� �����, ������� ���� (��� -- �� �������, ���� ��� � ������ ������� 20 ����� ����)\
#######################################

  
  
  
  
  
 