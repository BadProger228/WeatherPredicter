from Analizaton_Weather import WeatherPredicter
import pandas as pd

from mainForm import mainForm


predicter = WeatherPredicter()
predicter.analizate()
form = mainForm(predicter);
form.show();
