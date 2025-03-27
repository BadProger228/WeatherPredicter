import tkinter as tk
from Analizaton_Whether import Day, WhetherPredicter
import pandas as pd
from tkinter import ttk
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import threading

class mainForm:
    def __init__(self, whether_predict):
        self.day = None
        self.whether_predict = whether_predict;

    def calculate_weather_button(self):
        thread = threading.Thread(target=self.calculate_weather)
        thread.start()
   
    def calculate_weather(self):
        start_time = pd.to_datetime(self.start_time_entry.get())
        end_time = pd.to_datetime(self.end_time_entry.get())
        year_diff = end_time.year - start_time.year

        year_diff = end_time.year - start_time.year

        if year_diff >= 1:
            self.days = self.whether_predict.predict_whether_for_long_time(start_time, end_time, 12)
        else:
            self.days = self.whether_predict.predict_whether_for_long_time(start_time, end_time, 1)
        if self.listbox.get_children():
            self.listbox.delete(*self.listbox.get_children())
        for day in self.days:
            self.listbox.insert("", "end", values=(day.date.date(), f'Temerature: {int(day.get_min_temerature())} - {int(day.get_max_temerature())}'))
         
            
    
    
    def calculate_whether_for_next_day(self):
        self.day = self.whether_predict.predict_whether_for_long_time(datetime.now() + timedelta(days=1) ,datetime.now() + timedelta(days=2), 1)
        self.show_graphic()
        
    def show_graphic(self):
        
        found_day = None
        date = None
        selected_item = self.listbox.selection()
        if not selected_item and self.day == None:
            return;
        elif self.day != None:
            found_day = self.day[0]
            date = found_day.date.date()
        elif pd.to_datetime(self.end_time_entry.get()).year - pd.to_datetime(self.start_time_entry.get()).year >= 1:
            date = self.listbox.item(selected_item)['values'][0]
            found_day = self.whether_predict.predict_whether_for_long_time(pd.to_datetime(date), pd.to_datetime(date) + pd.Timedelta(days=1), 1)[0]
            
        else:
            date = self.listbox.item(selected_item)['values'][0]
            for day in self.days:
                if day.date.date() == pd.to_datetime(date).date():
                    found_day = day
                    break

        
        hours = list(range(24))
        temperature = found_day.predicts_for_temperature  

        
        plt.figure(figsize=(8, 6))  
        plt.plot(hours, temperature, label='Temperature', color='blue', marker='o')  

       
        feels_like_temperature = found_day.wind_chill_temperature()  
        plt.plot(hours, feels_like_temperature, label='Feel temperature', color='red', linestyle='--', marker='x')  

        
        plt.xlabel('Time (hour)')
        plt.ylabel('Temperature (*C)')
        plt.title(f'Temperature Graph for {date}')
        plt.grid(True)  
        plt.legend()  
        
        wind_speed = found_day.predicts_for_wind 
        plt.figure(figsize=(8, 6))  
        plt.plot(hours, wind_speed, label='Wind Speed', color='green', linestyle='-.', marker='s')  
        plt.xlabel('Time (hours)')
        plt.ylabel('Wind Speed (m/s)')
        plt.title('Wind Speed Data')
        plt.grid(True) 
        plt.legend()  

       
        plt.show()
        
        self.day = None
        

    
    def show(self):
        self.root = tk.Tk()
        self.root.title("Weather Predictor")
        self.root.geometry("420x400") 

        self.start_time_label = tk.Label(self.root, text="From what time? ")
        self.start_time_label.grid(row=0, column=0, padx=5, pady=5)
        self.start_time_entry = tk.Entry(self.root)
        self.start_time_entry.grid(row=0, column=1, padx=5, pady=5)

        self.end_time_label = tk.Label(self.root, text="To what time? ")
        self.end_time_label.grid(row=1, column=0, padx=5, pady=5)
        self.end_time_entry = tk.Entry(self.root)
        self.end_time_entry.grid(row=1, column=1, padx=5, pady=5)

        self.calculate_button = tk.Button(self.root, text="Find the Weather", command=self.calculate_weather_button)
        self.calculate_button.grid(row=2, column=0, columnspan=2, padx=5, pady=10)

        self.calculate_button_next_day = tk.Button(self.root, text="Find the Weather for Next Day", command=self.calculate_whether_for_next_day)
        self.calculate_button_next_day.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

        self.listbox = ttk.Treeview(self.root, columns=("Column 1", "Column 2"), show="headings")
        self.listbox.heading("Column 1", text="Date")
        self.listbox.heading("Column 2", text="Temperature")
        self.listbox.bind("<Double-1>", lambda event: self.show_graphic())
        self.listbox.grid(row=4, column=0, columnspan=2, padx=5, pady=10)

        self.root.mainloop()


