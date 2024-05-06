import customtkinter, tkinter
from cProfile import label
import tkinter
from tkinter import *
import pandas as  pd
import numpy as np
import csv
import string
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from nltk.corpus import stopwords
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn import tree
dsn=None
dic={}
flag=[]
ac1=[]
ac=[]
ac2=[]
ac3=[]
xKNN=[]
xLINREG=[]
xDECTREE=[]
xNAIVE=[]
lisreg=[]
dic1={}
ax=None
ay=None
az=None
customtkinter.set_appearance_mode("dark")  
customtkinter.set_default_color_theme("blue")
app = customtkinter.CTk()
def remove_punctuation_and_stopwords_and_lowercase(email):
    email_no_punct = [ch for ch in email if ch not in string.punctuation]
    email_no_punct = "".join(email_no_punct).split()
    email_no_punct_no_stopwords = [word.lower() for word in email_no_punct if word.lower() not in stopwords.words("english")]
    return email_no_punct_no_stopwords
def validation():
    label8=Label(app,text="validation",font=('Times New Roman','20'))
    label8.grid(column=0,row=7)
    label9=Label(app,text="Confusion Matrix",font=('Times New Roman','10'))
    label9.grid(column=0,row=8)
    button8=customtkinter.CTkButton(app, text="click",width=7,height=1,command=confusmtrx)
    button8.grid(column=1,row=8)
    label10=Label(app,text="Accuracy",font=('Times New Roman','10'))
    label10.grid(column=2,row=8)
    button9=customtkinter.CTkButton(app, text="click",width=7,height=1,command=acure)
    button9.grid(column=3,row=8)
###############################################################################################################
def navyby():
    global dsn
    global flag
    global ac2
    flag.append(3)
    dsn=dsname()
    df= pd.read_csv(dsn, encoding='latin-1')
    y=df['v1']
    x=df['v2']
    x.apply(remove_punctuation_and_stopwords_and_lowercase)
    x=pd.get_dummies(data=x)
    y = y.map({'spam': 1, 'ham':0})
    import matplotlib.pylab as plt
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    from sklearn.naive_bayes import GaussianNB
    nBmodel=GaussianNB()
    nBmodel.fit(x_train,y_train)
    y_predicted=nBmodel.predict(x_test)
    from sklearn import metrics
    validation()
    ac2=[metrics.confusion_matrix(y_test, y_predicted), metrics.accuracy_score(y_predicted,y_test)]

###############################################################################################################
def KNN():
    global flag
    global ac1
    flag.append(1)
    x=dsname()
    df= pd.read_csv(x, encoding='latin-1')
    y=df['v1']
    x=df['v2']
    x.apply(remove_punctuation_and_stopwords_and_lowercase)
    y=y.map({'spam': 1, 'ham':0})
    x=pd.get_dummies(data= x)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train,y_train)
    pred=knn.predict(x_test)
    from sklearn.metrics import classification_report,confusion_matrix
    validation()
    ac1=[confusion_matrix(y_test,pred), metrics.accuracy_score(pred,y_test)]
###############################################################################################################
def DecTree():
    global flag
    global ac
    flag.append(2)
    x=dsname()
    df= pd.read_csv(x, encoding='latin-1')
    y=df['v1']
    x=df['v2']
    x.apply(remove_punctuation_and_stopwords_and_lowercase)
    y=y.map({'spam': 1, 'ham':0})
    x=pd.get_dummies(data= x)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    model=tree.DecisionTreeClassifier()
    model.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    from sklearn.metrics import accuracy_score , confusion_matrix
    validation()
    ac= [confusion_matrix(y_test,y_predict),accuracy_score(y_test,y_predict)]
###############################################################################################################
def LinReg():
    global flag
    global ac3
    global lisreg
    global ax
    global az
    global ay
    flag.append(4)
    x=dsname()
    df= pd.read_csv(x, encoding='latin-1')
    y=df['v1']
    x=df['v2']
    x.apply(remove_punctuation_and_stopwords_and_lowercase)
    y=y.map({'spam': 1, 'ham':0})
    x=pd.get_dummies(data= x)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    LRModel = LinearRegression()
    LRModel.fit(x_train, y_train)
    prediction = LRModel.predict(x_test)
    ax=[metrics.mean_absolute_error(y_test,prediction)]
    ay=[metrics.mean_squared_error(y_test,prediction)]
    az=[np.sqrt(metrics.mean_squared_error(y_test,prediction))]
    validation()
###############################################################################################################
def Classification():
    global app
    label4=Label(app,text="KNN",font=('Times New Roman','10'))
    label4.grid(column=3,row=3)
    button4=customtkinter.CTkButton(app, text="click",width=7,height=1,command=KNN)
    button4.grid(column=4,row=3)
    label5=Label(app,text="Decision Tree ",font=('Times New Roman','10'))
    label5.grid(column=3,row=4)
    button5=customtkinter.CTkButton(app, text="click",width=7,height=1,command=DecTree)
    button5.grid(column=4,row=4)
    label6=Label(app,text="Naïve Bayes",font=('Times New Roman','10'))
    label6.grid(column=3,row=5)
    button6=customtkinter.CTkButton(app, text="click",width=7,height=1,command=navyby)
    button6.grid(column=4,row=5)
    ##############################################################################################################
def Regression():
    label7=Label(app,text="Linear Regression",font=('Times New Roman','10'))
    label7.grid(column=0,row=3)
    button7=customtkinter.CTkButton(app, text="click",width=7,height=1,command=LinReg)
    button7.grid(column=1,row=3)
##############################################################################################################
def dsname():
    global entrybox1
    x=str(entrybox1.get())
    return x
##############################################################################################################
def confusmtrx():
    global flag
    global xKNN
    global xLINREG
    global xDECTREE
    global xNAIVE
    if 1 in flag:
        xKNN=ac1[0]
        la1=Label(app,text=f"{xKNN} KNN")
        la1.grid(row=10,column=0)
    if 2 in flag :
        xDECTREE=ac[0]
        la2=Label(app,text=f"{xDECTREE} decision tree")
        la2.grid(row=11,column=0)
    if 3 in flag:
        xNAIVE=ac2[0]
        la3=Label(app,text=f"{xNAIVE} naive")
        la3.grid(row=12,column=0)
############################################################################################################## 
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
data = pd.read_csv('spam.csv' , encoding='latin-1')
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis =1)
data = data.rename(columns={"v1":"label", "v2":"email"})
data['spam'] = data['label'].map({'spam': 1, 'ham':0})
data['email'].apply(remove_punctuation_and_stopwords_and_lowercase).head()
from collections import Counter
data_ham  = data[data['spam'] == 0].copy()
data_spam = data[data['spam'] == 1].copy()
data_ham.loc[:, 'email'] = data_ham['email'].apply(remove_punctuation_and_stopwords_and_lowercase)
words_data_ham = data_ham['email'].tolist()
data_spam.loc[:, 'email'] = data_spam['email'].apply(remove_punctuation_and_stopwords_and_lowercase)
words_data_spam = data_spam['email'].tolist()
list_ham_words = []
for sublist in words_data_ham:
    for item in sublist:
        list_ham_words.append(item)
list_spam_words = []
for sublist in words_data_spam:
    for item in sublist:
        list_spam_words.append(item)
c_ham  = Counter(list_ham_words)
c_spam = Counter(list_spam_words)
cham = int(0.4*len(c_ham))
cspam = int( 0.4*len(c_spam))
df_hamwords_top40percent  = pd.DataFrame(c_ham.most_common(cham),  columns=['word', 'count'])
df_spamwords_top40percent = pd.DataFrame(c_spam.most_common(cspam), columns=['word', 'count'])
###############################################################################################################
def acure():
    global dic1
    global dic
    global flag
    global ax
    global ay
    global az
    if 1 in flag:
        dic["KNN"]=[ac1[1]]
    if 2 in flag :
        dic["dec tree"]=[ac[1]]
    if 3 in flag:
        dic["niavyby"]=[ac2[1]]
    if 4 in flag:
        dic1["MEAN ABS ERROR"]=ax
        dic1["MEAN SQUARE ERROR"]=ay
        dic1["MEAN root ERROR"]=az
        pf2=pd.DataFrame(data=dic1)
        fig2=pf2.plot.bar().get_figure();
        bar2 =FigureCanvasTkAgg(fig2,app)
        bar2.get_tk_widget().grid(column=9,row=13)
    pf=pd.DataFrame(data=dic)
    fig=pf.plot.bar().get_figure();
    bar1 =FigureCanvasTkAgg(fig,app)
    bar1.get_tk_widget().grid(column=5,row=13)
############################################################################################################### 
app.geometry("1280x720")
label1=Label(app,text="email spam detection",font=('Times New Roman','20'))
label1.grid(column=3,row=0)
entrybox1=Entry()
entrybox1.grid(column=0,row=1)
button1=customtkinter.CTkButton(app, text="Select DS", width=7,height=1,command=dsname)
button1.grid(column=1,row=1)
label2=Label(app,text="Regression ",font=('Times New Roman','10'))
label2.grid(column=0,row=2)
button2=customtkinter.CTkButton(app, text="click",width=7,height=1,command=Regression)
button2.grid(column=1,row=2)
label3=Label(app,text="Classification ",font=('Times New Roman','10'))
label3.grid(column=3,row=2)
button3=customtkinter.CTkButton(app, text="click",width=7,height=1,command=Classification)
button3.grid(column=4,row=2)
app.mainloop()