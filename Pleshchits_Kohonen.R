install.packages('DAAG')
library('DAAG')

data('spam7')


spam <- spam7[,-7]
yesno <- spam7[,7]

library('kohonen')
som.spam <- som(scale(spam), 
                 grid = somgrid(5, 5, 'hexagonal'))
plot(som.spam, main = 'spam Kohonen')

plot(som.spam, type = 'changes', main = 'Changes')

4601*0.8
4601-3681

train <- sample(nrow(spam), 3681) #бесповторная выборка
train

x_train <- scale(spam[train,]) 
x_test <- scale(spam[-train,], 
                center = attr(x_train, 'scaled:center'),
                scale = attr(x_train, 'scaled:center')) #шкалируем данные по предыдущим данным: по предыдущей выборке

train_data <- list(measurements = x_train,
                   yesno = yesno[train])
test_data <- list(measurements = x_test,
                  yesno = yesno[-train])

#карта, которую мы будем обычать
som.grid <- somgrid(5, 5, 'hexagonal') #шаблон карты
som.spam <- supersom(train_data, grid = som.grid)

plot(som.spam)

som_predict <- predict(som.spam, newdata = test_data) #вносим тестовую выборку в обученную НС
table(yesno[-train], som_predict$predictions[['yesno']]) #матрица совпадений

classif <-  data.frame(yesno = yesno[train], class = som.spam[["unit.classif"]])
classif
