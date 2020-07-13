import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


def main():
    st.title("Aplicativo web de classificacao binaria")
    st.sidebar.title("Aplicativo web de classificacao binaria")
    st.markdown("Seus cogumelos sao comestiveis ou venenosos? 🍄")
    st.sidebar.markdown("Seus cogumelos sao comestiveis ou venenosos? 🍄")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("./mushrooms.csv")
        labelencoder = LabelEncoder()
        for col in data.columns:
            data[col] = labelencoder.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Matriz de confusao' in metrics_list:
            st.subheader("Matriz de confusao")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'Curva ROC' in metrics_list:
            st.subheader("Curva ROC")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Curva Precisao-Recall' in metrics_list:
            st.subheader('Curva Precisao-Recall')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    class_names = ['edible', 'poisonous']

    x_train, x_test, y_train, y_test = split(df)

    st.sidebar.subheader("Escolha o Classificador")
    classifier = st.sidebar.selectbox("Classificador",
                                      ("Maquina de Suporte de Vetores", "Regressao Logistica", "Floresta Randomica"))

    if classifier == 'Maquina de Suporte de Vetores':
        st.sidebar.subheader("Hiperparametros do modelo")
        # parameters
        C = st.sidebar.number_input("C (Parametro de regularizacao)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Coeficiente do Kernel)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("Qual metrica voce quer plotar?",
                                         ('Matriz de confusao', 'Curva ROC', 'Curva Precisao-Recall'))

        if st.sidebar.button("Classifique", key='Classificar'):
            st.subheader("Resultados da Maquina de Suporte de Vetores (SVM) ")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Acuracia: ", accuracy.round(2))
            st.write("Precisao: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Regressao Logistica':
        st.sidebar.subheader("Hiperparametros do modelo")
        C = st.sidebar.number_input("C (Parametro de regularizacao)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Numero maximo de iteracoes", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("Quais metricas voce quer plotar?",
                                         ('Matriz de confusao', 'Curva ROC', 'Curva Precisao-Recall '))

        if st.sidebar.button("Classificar", key='Classificar'):
            st.subheader("Regressao Logistica Resultados")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Acuracia: ", accuracy.round(2))
            st.write("Precisao: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Floresta Randomica':
        st.sidebar.subheader("Hiperparametros do modelo")
        n_estimators = st.sidebar.number_input("O numero de arvores na floresta", 100, 5000, step=10,
                                               key='n_estimators')
        max_depth = st.sidebar.number_input("O tamanho maximo da arvore", 1, 20, step=1, key='n_estimators')
        bootstrap = st.sidebar.radio("Amostras em bootstrap quando construir as arvores", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("Que metricas plotar?",
                                         ('Matriz de confusao', 'Curva ROC', 'Curva Precisao-Recall'))

        if st.sidebar.button("Classificar", key='Classificar'):
            st.subheader("Resultados da Floresta Randomica")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
                                           n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Acuracia: ", accuracy.round(2))
            st.write("Precisao: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Mostrar os dados", False):
        st.subheader("Classificacao de uma base de dados de cogumelos")
        st.write(df)
        st.markdown(
            "Essa [base dados](https://archive.ics.uci.edu/ml/datasets/Mushroom) inclui demonstrações de amostras hipoteticas correspondente a 23 especies de cogumelos com guelra "
            "Na familia Agaricus and Lepiota (pp. 500-525). Cada especie é identificada como definitivamente comestivel, ou definitivamente venenosa, "
            "ou se não se sabe se é comestivel. Essa aula os dados foram combinados com o de cogumelos venenosos. Pagina criada por Israel Andrade")


if __name__ == '__main__':
    main()
