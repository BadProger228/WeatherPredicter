from Analizaton_Whether import WhetherPredicter
import pandas as pd

from mainForm import mainForm


predicter = WhetherPredicter()
predicter.analizate()
form = mainForm(predicter);
form.show();
