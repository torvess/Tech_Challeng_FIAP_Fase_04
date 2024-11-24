from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
import pandas as pd

# classe para instanciar o modelo
class ModelSeasonalNaive():
    def __init__(self, train_data, h):
        self.train_data = train_data
        self.h = h

    # treinamento a aplicacao dos modelos
    def fit_SeasonalNaive(self):
        self.model_SeasonalNaive = StatsForecast(models=[SeasonalNaive(season_length=7)], freq='D', n_jobs=-1)
        self.model_SeasonalNaive.fit(self.train_data)
        print('Treinamento concluído')

    def predic_SeasonalNaive(self, date):
        self.h = (pd.Timestamp(date) - self.train_data['ds'].max()).days
        forecast = self.model_SeasonalNaive.predict(h=self.h, level=[90])
        return  forecast[forecast['ds'] == date]['SeasonalNaive'].values[0]