library('keras')
library('tensorflow')
library('devtools')
library('usethis')

# Устанавливаем Keras
install_keras()
#откат версии

#####################
#при обишке "AttributeError: module 'tensorflow' has no attribute 'VERSION'"
#откатить на версию: 1.13.1
tensorflow::install_tensorflow(version = "1.13.1")

# загрузка данных
mnist <- dataset_mnist()
# разбиваем данные на 4 объекта
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

#####################
#архитектура нейросети
network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = c(28*28)) %>%
  layer_dense(units = 10, activation = 'softmax')
# Добавляем для нейронной сети оптимизатор, функцию потерь, 
#какие метрики выводить на экран
network %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy'))
#####################
# Изначально массивы имеют размерность 60000, 28, 28, сами значения изменяются в пределах от 0 до 255
# для обучения нейронной сети потребуется преобразовать форму 60000, 28*28, а значения перевести в размерность от 0 до 1

#меняем размерность:
train_images <- array_reshape(train_images, c(60000, 28*28))
#меняем область значений
train_images <- train_images/255
str(train_images)

#повторяем для тестовой
test_images <- array_reshape(test_images, c(10000, 28*28))
test_images <- test_images/255

# создаем категории для ярлыков
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)


#тренировка НС
network %>% fit(train_images, train_labels, epochs = 10, batch_size = 128)
# точность составила .9971 (liss = .01)

metric <- network %>% evaluate(test_images, test_labels)
metric
# по тестоовой выборке -- .9821 (loss = .0665)

##предсказываем наблюдения
#выборка по тестовым наблюдениям (первые 10; последний 10)
a_test <- rbind (test_images[1:10,], test_images[9991:10000,])

predict <- network %>% predict_classes(a_test)
test_labels1 <- mnist$test$y
a_test_lables <- rbind (test_labels1[1:10], test_labels1[9991:10000])        
trans <- t(a_test_lables)
data <- as.numeric(t) 

#сравним полученные и расчетные данные
compare <- cbind(data, predict)
compare

# обучение сетей с добавлением валидации (появляется интерактивны график с точками каждой эпохи)
history <- network %>% fit(train_images, train_labels,
                           epochs = 10, batch_size = 128,
                           validation_split = 0.2)

########################
#нарисуем числа из массива
par(mfrow = c(3, 3))
#1
a1 <- mnist$test$x[9991, 1:28, 1:28]
a1
image(as.matrix(a1))
#2
a2 <- mnist$test$x[9992, 1:28, 1:28]
image(as.matrix(a2))
#3
a3 <- mnist$test$x[9993, 1:28, 1:28]
image(as.matrix(a3))
#4
a4 <- mnist$test$x[9994, 1:28, 1:28]
image(as.matrix(a4))
#5
a5 <- mnist$test$x[9995, 1:28, 1:28]
image(as.matrix(a5))
#6
a6 <- mnist$test$x[9996, 1:28, 1:28]
image(as.matrix(a6))
#7
a7 <- mnist$test$x[9997, 1:28, 1:28]
image(as.matrix(a7))
#8
a8 <- mnist$test$x[9998, 1:28, 1:28]
image(as.matrix(a8))
#9
a9 <- mnist$test$x[9999, 1:28, 1:28]
image(as.matrix(a9))
par(mfrow = c(1, 1))
#10 (изгой)
a10 <- mnist$test$x[10000, 1:28, 1:28]
image(as.matrix(a10))  











