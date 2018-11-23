#ТЯЖЕЛЫЙ СЛУЧАЙ
#x = np.arange(length, dtype=float).reshape((length, 1))
#y = x + (np.random.rand(length)*10).reshape((length, 1))
# Загружаем библиотеки
import numpy as np # работа с векторами
import matplotlib.pyplot as plt # рисовать графики
import pandas as pd # для работы с матрицами
import os as os #для смены директории
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Установим директорию для загрузки данных для модели
#os.chdir('C:/папка с файлом')

# загружаем данные
dataset = pd.read_csv('C:\\Salary_Data.csv')



# создаем Y переменную = заработные платы и Х перменые, в нашем случае только
# один набор = опыт
# по другому Y называют зависимой переменной а Х контролирующей
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Разделим данные на тренировочную и тестовую выборку
# random_state = позволяет получать всегда одинаковое разбиение выборки
# X пишем сбольшой буквы это вектор.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Тренировка линейной регрессии на тренировочном наборе данных


#Создаем регрессора как метод, не указываем дополнительные параметры
regressor = LinearRegression()
#Тренеруем модель на тренировочных данных.
regressor.fit(X_train,y_train) # ваша первая модель машинного обучения!

#Прогнозируем результаты тестовой выборки
#Разница между y_pred и y_test в том, что y_pred это прогнозные значения
#Теперь мы можем сравнить их с тестовыми значениями
y_pred = regressor.predict(X_test)

#Визуализация результатов тестового набор данных
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Заработная плата vs Опыт(Тренировочные данные)')
plt.xlabel('Опыт в годах')
plt.ylabel("Заработная плата")
plt.show() #команда для отображения графика в python. Убрать комментарий вначале строки, чтобы увидеть график.
plt.savefig('graph.png',bbox_inches='tight') # технический трюк


#Визуализация результатов тестового набор данных
plt.scatter(X_test,y_test,color = 'red')
#линию регресии не меняем. Мы получим тестовые и оценим как линия регрессии
# описывает тестовый набор
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Заработная плата vs Опыт(Тренировочные данные)')
plt.xlabel('Опыт в годах')
plt.ylabel("Заработная плата")
plt.show


x = np.arange(length, dtype=float).reshape((length, 1))
y = x + (np.random.rand(length)*10).reshape((length, 1))