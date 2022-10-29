from flask import Flask, request, render_template
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle

app = Flask(__name__, template_folder='templates')

# Функция для извлечения признаков из полей формы
def get_data(features, params):
    param_names = features.keys()
    data = dict.fromkeys(param_names, None)
    err = ''
    # Преобразуем значения из строкового в числовой тип
    for param_name, param_value in params.items():
        if param_value.strip(' \t') != '':
            try:
                data[param_name] = float(param_value)
            except:
                err += f'{features[param_name]} - некорректное значение "{param_value}"\n'
    return dict(zip(features.values(), data.values())), err

@app.route('/', methods=['POST', 'GET'])
def model3_NN4():
    # Признаки для прогнозирования
    features = {
        'p2': 'Плотность, кг/м3',
        'p3': 'модуль упругости, ГПа',
        'p4': 'Количество отвердителя, м.%',
        'p5': 'Содержание эпоксидных групп,%_2',
        'p6': 'Температура вспышки, С_2',
        'p7': 'Поверхностная плотность, г/м2',
        'p8': 'Модуль упругости при растяжении, ГПа',
        'p9': 'Прочность при растяжении, МПа',
        'p10': 'Потребление смолы, г/м2',
        'p11': 'Угол нашивки, град',
        'p12': 'Шаг нашивки',
        'p13': 'Плотность нашивки'
        }
    # Переменные для формы
    params = {
        'p2': '',
        'p3': '', 
        'p4': '', 
        'p5': '', 
        'p6': '', 
        'p7': '',
        'p8': '', 
        'p9': '', 
        'p10': '', 
        'p11': '', 
        'p12': '', 
        'p13': ''
        }
    err = ''
    X = pd.DataFrame()
    p1 = ''
    # Получены данные из полей формы
    if request.method == 'POST':
        params = request.form.to_dict()
        data, err = get_data(features, params)
        if err == '':
            X = pd.DataFrame(data, index=[0])
            filename_pre3 = 'preprocessor3'
            file = open(filename_pre3, 'rb')
            pre3 = pickle.load(file)
            file.close()
            model3_NN4 = tf.keras.models.load_model('model3_NN4/')
            X3 = pre3.transform(X)
            y3 = model3_NN4.predict(X3)
            p1 = y3[0]

    # Выводим результаты
    return render_template('model3_NN4.html', params=params, err=err, inputs=X.to_html(), p1=p1)

app.run()