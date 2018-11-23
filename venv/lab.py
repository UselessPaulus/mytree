import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#sklearn.datasets.open_diabetes

#Загружаем базу данных больных диабетом
diabetes = datasets.load_diabetes()


#Помещаем её только в одну функцию
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Разделить данные на обучающие / тестовые выборки
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Разделить цели на обучающие / тестовые выборки
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Создать объект линейной регрессии
regr = linear_model.LinearRegression()

# Обучить модель, используя обучающие наборы
regr.fit(diabetes_X_train, diabetes_y_train)

# Сделать прогнозы с помощью набора тестов
diabetes_y_pred = regr.predict(diabetes_X_test)

# Коэффициенты
print('Коэффициенты: \n', regr.coef_)
# Средняя квадратичная ошибка
print("Средняя квадратичная ошибка: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Оценка дисперсии на 1 - идеальное предсказание
print('Разница: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Выводы
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()