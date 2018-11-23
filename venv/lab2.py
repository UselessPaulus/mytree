## -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import subprocess
import Graphviz
import nltk
from IPython.display import Image
from sklearn import tree
import pydot
from Graphviz import Source
import pydotplus
from io import StringIO
import pydot
import pyparsing
import os
import PyQt5
from ete3 import Tree
from ete3 import TreeStyle
from IPython.display import display
from nltk.metrics import *
import collections
os.environ["PATH"] += os.pathsep + 'C:\\Users\ganch\PycharmProjects\Clastor\venv\Lib\site-packages\Graphviz\bin'
#Dataframe (индексирование многомерного массива)
df = pd.io.excel.read_excel('C:\\session.xlsx', sheet_name='Лист1')

print(df.head(10))

#Создаю матрицу признаков и результирующий столбец

X = df.drop('Session', axis=1)
y = df['Session']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
clf = tree.DecisionTreeClassifier(max_depth=23)

clf.fit(X_train, y_train)

clf.score(X_test, y_test)

test = [[74, 10,20]]
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




#graph = pydotplus.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png")

#pydot.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png")


#raph = pydotplus.graph_from_dot_data(clf(23))
#Image(graph.create_png())

#png_bytes = graph.pipe(format='png')
#with open('dtree_pipe.png','wb') as f:
 #   f.write(png_bytes)

#Image(png_bytes)











#(graph,) = pydot.graph_from_dot_file('somefile.dot')
#graph.write_png('somefile.png')
#dot_data = tree.export_graphviz(clf, out_file=None)
#graph = graphviz.Source(dot_data)
#graph.render("Session")


#tree.export_graphviz(tree,out_file='tree.dot')

#export_graphviz(tree, out_file='C:\\Program Files\att\att.dot',feature_names=X.columns,filled=True)
#tree.export_graphviz(clf,out_file='tree.dot')
#subprocess.call(['dot', '-Tpng tree.dot -o tree.png'])

#dot_data = tree.export_graphviz(my_tree_one, out_file='tree.dot')
#graph = pydotplus.graph_from_dot_data(dot_data)
#Image(graph.create_png())
#export_graphviz(tree, out_file='C:\\Program Files (x86)tree_att.dot',feature_names=x.columns,filled=True)

#tree.export_graphviz(dtreg, out_file='tree.dot')
#dotfile = StringIO()
#tree.export_graphviz(dtreg, out_file=dotfile)
#pydot.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png")

#export_graphviz(tree, out_file='pics/tree_att.dot',feature_names=X.columns,filled=True)
#system("dot -Tpng pics/tree_att.dot -o pics/tree_att.png")
#dot -Tpng pics/tree_att.dot -o pics/tree_att.png
#<img scr ='pics/tree_att.png'>