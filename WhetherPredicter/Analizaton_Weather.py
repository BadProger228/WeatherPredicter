import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import joblib
import os


class Day:
    def __init__(self, date, predicts):
        self.date = date;
        self.whether_info = None
        self.predicts_for_temperature = []
        self.predicts_for_precipitation = []
        self.predicts_for_wind = []
        self.predicts_for_cloud_cover = []
        
        for val in predicts:
            self.predicts_for_temperature.append(val[0][0])  

        for val in predicts:
            self.predicts_for_precipitation.append(val[0][1])
        
        for val in predicts:
            self.predicts_for_wind.append(val[0][2])
        
        for val in predicts:
            self.predicts_for_cloud_cover.append(val[0][3])

    
    def get_info_for_time(self, hour):
        return self.predicts_for_day[hour]
    
    def get_max_temerature(self):
        return max(self.predicts_for_temperature)
    
    def get_min_temerature(self):
        return min(self.predicts_for_temperature)
    
    def wind_chill_temperature(self):
        result = []
        for i in range(24):
            temperature = self.predicts_for_temperature[i] 
            wind_speed = self.predicts_for_wind[i] * 1000 / 3600
            result.append(35.74 + 0.6215 * temperature - 35.75 * (wind_speed ** 0.16) + 0.4275 * temperature * (wind_speed ** 0.16))
        return result
        
class WeatherPredicter:
    def __init__(self, path_to_table=None):
        self.path_to_table = path_to_table
        self.model = None
        self.scaler = None

        if path_to_table is None:
            custom_objects = {'mse': tf.keras.metrics.MeanSquaredError()}
            path_to_file = os.getcwd()
            self.model = tf.keras.models.load_model(f'{path_to_file}\my_saved_model.keras', custom_objects=custom_objects)
            self.scaler = joblib.load(f'{path_to_file}\scaler_model.pkl')

    def preprocess_data(self, data):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['year'] = data['timestamp'].dt.year
        data['month'] = data['timestamp'].dt.month
        data['day'] = data['timestamp'].dt.day
        data['hour'] = data['timestamp'].dt.hour
        data = data.drop(columns=['timestamp'])
        return data
    
    def build_model(self, input_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4)  # 4 выходных нейрона для предсказания 4 параметров погоды
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def analizate(self):
        
        if(self.model is not None):
            return;
        
        weather_data = pd.read_excel(self.path_to_table)

        weather_data = self.preprocess_data(weather_data)

        y = weather_data[['Basel Temperature [2 m elevation corrected]', 'Basel Precipitation Total', 'Basel Wind Gust', 'Basel Cloud Cover Total']]

        X = weather_data.drop(columns=y.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        model = self.build_model(X_train_scaled.shape[1])

        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)
        
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        model.save('my_saved_model.keras')

        self.model = model
        model.save('my_model.h5')
        joblib.dump(self.scaler, fr'{os.getcwd()}\scaler_model.pkl')
    
    def predict_weather(self, full_date):
        input_data = pd.DataFrame({'timestamp': [full_date]})
        input_data = self.preprocess_data(input_data)

        input_data_scaled = self.scaler.transform(input_data)

        predicted_weather = self.model.predict(input_data_scaled)
        
        return predicted_weather
    
    def predict_weather_for_long_time(self, start_date, end_date, dx):
        
        start_date = start_date.replace(hour=0, minute=0, second=0)
        end_date = end_date.replace(hour=0, minute=0, second=0)

        result = []
        days = []
        info_for_day = []
        changet_date = start_date;
        while start_date.date() < end_date.date():
               
             info_for_day.append(self.predict_weather(start_date))
             start_date += pd.Timedelta(hours=dx)
             
             if(changet_date.date() != start_date.date()):
                 days.append(Day(changet_date, info_for_day))
                 changet_date = start_date
                 info_for_day = []
                 
        return days


    