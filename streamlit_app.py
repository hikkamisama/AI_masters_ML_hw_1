from itertools import combinations
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import streamlit as st

eps = 1e-4

class CustomFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, include_square=True, include_inverse=True,
                 include_log=True, include_interactions=True):
        self.include_square = include_square
        self.include_inverse = include_inverse
        self.include_log = include_log
        self.include_interactions = include_interactions
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        X_new = X.copy()

        # x^2
        if self.include_square:
            for c in self.feature_names_in_:
                X_new[f"{c}__sq"] = X[c] ** 2

        # 1/x -
        if self.include_inverse:
            for c in self.feature_names_in_:
                X_new[f"{c}__inv"] = 1 / (X[c] + eps)

        # log(x)
        if self.include_log:
            for c in self.feature_names_in_:
                X_new[f"{c}__log"] = np.log1p(X[c])

        # x_i * x_j
        if self.include_interactions:
            for c1, c2 in combinations(self.feature_names_in_, 2):
                X_new[f"{c1}__mul__{c2}"] = X[c1] * X[c2]

        return X_new

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.fit_transform(pd.DataFrame(
            np.zeros((1, len(self.feature_names_in_))),
            columns=self.feature_names_in_
        )).columns))

@st.cache_resource
def load_model():
    with open('models/linear_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

st.title("Модель предсказания цены машины на основе различных признаков")

model = load_model()

st.header("Ввод данных") 

categoric = ['fuel', 'seller_type', 'transmission', 'owner', 'car_brand', 'seats_int']
numeric = ['mileage_float', 'max_power_float', 'torque_float', 'max_torque_rpm', 'engine_int', 'year', 'km_driven']

car_brand = st.text_input("Введите бренд машины")
fuel = st.selectbox("Выберите тип топлива", ["Diesel", "Petrol", "CNG", "LPG", "Other"])
seller_type = st.selectbox("Выберите тип продавца", ["Individual", "Dealer", "Trustmark Dealer", "Other"])
transmission = st.selectbox("Выберите тип коробки передач", ["Manual", "Automatic", "Other"])
owner = st.selectbox("Выберите категорию владельца", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car", "Other"])
seats_int = st.number_input("Введите число сидений", min_value=0)

mileage_float = st.number_input(label="Введите потребление топлива (в км/л)",format="%.2f")
max_power_float = st.number_input(label="Введите максимальную мощность (в лошадиных силах)",format="%.2f")
torque_float = st.number_input(label="Введите момент силы (в Нм)",format="%.2f")
max_torque_rpm = st.number_input(label="Введите максимальное число оборотов (в оборот/мин)", min_value=0)
engine_int = st.number_input(label="Введите объем цилиндра (в кубических сантиметрах)", min_value=0)
year = st.number_input(label="Введите год производства", min_value=0)
km_driven = st.number_input(label="Введите километраж", min_value=0)

inputs = {
    "car_brand": car_brand,
    "fuel": fuel,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner,
    "seats_int": seats_int,
    "mileage_float": mileage_float,
    "max_power_float": max_power_float,
    "torque_float": torque_float,
    "max_torque_rpm": max_torque_rpm,
    "engine_int": engine_int,
    "year": year,
    "km_driven": km_driven
}

if st.button("Предсказать цену"):
    df = pd.DataFrame([inputs])
    pred = model.predict(df)
    
    if pred.shape[0] > 1:
        formatted = [f"{elem:.1f}" for elem in pred.tolist()]
        st.metric("Предсказанные цены:", f"{formatted}")
    else:
        st.metric("Предсказанная цена:", f"{pred[0]:.1f}")

uploaded_file = st.file_uploader("Загрузите CSV c данными", type=["csv"])
if uploaded_file:
    if st.button("Предсказать цену"):
        df = load_data(uploaded_file)

        pred = model.predict(df)
        
        if pred.shape[0] > 1:
            formatted = [f"{elem:.1f}" for elem in pred.tolist()]
            st.metric("Предсказанные цены:", f"{formatted}")
        else:
            st.metric("Предсказанная цена:", f"{pred[0]:.1f}")


st.header("Графики")

st.image("plots/pairplot.png", caption="Исследование взаимодействий численных признаков")

st.image("plots/feature_corrs.png", caption="Корреляции Пирсона численных признаков")

st.image("plots/categoric.png", caption="Исследование целевой переменной в зависимости от категориальных признаков")

st.image("plots/feature_importance.png", caption="Топ-20 признаков по весу")

