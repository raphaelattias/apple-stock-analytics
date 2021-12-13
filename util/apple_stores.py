from tqdm.autonotebook import tqdm
from geopy.geocoders import Nominatim
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import numpy as np

def fig_world():
  df = pd.read_pickle('data/apple_stores/stores.pkl')
  fig = go.Figure(go.Scattergeo(lat=df['latitude'],
                                lon=df['longitude'], 
                                text=df['location'],
                                marker = dict(color = 'rgb(30, 50, 170)')
                                )
  )
  fig.update_geos(projection_scale=0.8)
  #fig.update_layout(height=800, margin={"r":0,"t":0,"l":0,"b":0})
  fig['data'][0]['showlegend'] = True
  fig['data'][0]['name']='Apple Store Location'

  df['name'] = "Apple Store Location"
  fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
          title_text='Apple Store locations worldwide',
          title_x=0.5,
          title_y = 0.98,
          geo=dict(
              projection_type='orthographic',
              center_lon=0,
              center_lat=0,
              projection_rotation_lon=0,
              showland=True,
              showcountries=True,
              landcolor='rgb(243, 243, 243)',
              countrycolor='rgb(170, 170, 170)'
          ),
        updatemenus=[dict(type='buttons', showactive=True,
                                  y=0.95,
                                  x=0.1,
                                  xanchor='right',
                                  yanchor='top',
                                  pad=dict(t=0, r=10),
                                  buttons=[dict(label='▶️',
                                                method='animate',
                                                args=[None, 
                                                      dict(frame=dict(duration=1, 
                                                                      redraw=True),
                                                          transition=dict(duration=0),
                                                          fromcurrent=True,
                                                          mode='immediate')
                                                    ]),
                                                    dict(label='⏸',
                                                method='animate',
                                                args=[None, 
                                                      dict(frame=dict(duration=0, 
                                                                      redraw=False),
                                                          transition=dict(duration=0),
                                                          fromcurrent=True,
                                                          mode='immediate')
                                                    ])])]
      )
  lon_range = np.arange(0, 360, 1)

  frames = [go.Frame(layout=dict(geo_center_lon=lon,
                                geo_projection_rotation_lon =lon
                            )) for lon in lon_range]
  fig.update_layout(legend=dict(
      yanchor="bottom",
      y=0.9,
      xanchor="left",
      x=0.85
  ))

  fig.update(frames=frames)
  fig.write_html('figures/world_map_apple_stores.html')
  print(f'Map of apples stores saved in figures/world_map_apple_stores.html')
  fig.show()


def find_stores():
  geolocator = Nominatim(user_agent='raphael.attias@outlook.com')
  lat = []
  lon = []
  locations = []
  countries = ['fr','chfr','es','de','au', 'at','be','bz','ca','cn','kr','ae', 'hk','it','jp','mo','mx','nl', 'sg','se','tw','th','tr']
  countries = ['/'+i+'/' for i in countries]
  countries.append('/')
  print(countries)

  for country in countries:
    URL = f"https://www.apple.com{country}retail/storelist/"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="page-store-list")
    try:
      stores = results.find_all("div", class_="store-address")
      for store in tqdm(stores):
        store = str(store)
        if country == "jp":
          city = store.split('/retail/')[1].split('/')[0]
        else:
          city = store.split('<span>')[1].split('<!')[0]
        
        try:
          location = geolocator.geocode(city)
          #print((location.latitude, location.longitude), location.address.split(', ')[0])
          lat.append(location.latitude)
          lon.append(location.longitude)
          locations.append(location.address.split(', ')[0])
        except:
          pass
    except:
      pass

  stores = pd.DataFrame({'location':locations,'longitude':lon, 'latitude':lat})
  stores.to_pickle(r'data/apple_stores/stores.pkl')
  print(f'Dictionnary of apples stores saved in data/apple_stores/stores.pkl')