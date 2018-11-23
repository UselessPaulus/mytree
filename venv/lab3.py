## -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import subprocess
import Graphviz
from sklearn import tree
import pydot
from Graphviz import Source
import pydotplus
from io import StringIO
import pydot
import pyparsing
import os

os.environ["PATH"] += os.pathsep + 'C:\\Users\ganch\PycharmProjects\Clastor\venv\Lib\site-packages\Graphviz\bin'
#Dataframe (индексирование многомерного массива)
df = pd.io.excel.read_excel('C:\\lgtu.xlsx', sheet_name='Лист1')

print(df.head(10))

#Создаю матрицу признаков и результирующий столбец

X = df.drop('Decision', axis=1)
y = df['Decision']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
clf = tree.DecisionTreeClassifier(max_depth=23)

clf.fit(X_train, y_train)

clf.score(X_test, y_test)

test = [[200, 53,53]]
predicted = clf.predict(test)


print(predicted)





#graph = Source(tree.export_graphviz(clf, out_file=None
#   , class_names=['0', '1', '2']
#   , filled = True))
#display(PNG(graph.pipe(format='png')))

dotfile = StringIO()
tree.export_graphviz(clf,out_file=dotfile)
graph=pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_png("dtree.png")
