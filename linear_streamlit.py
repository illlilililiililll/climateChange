import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from Linearregression import predict_temperature_change
from PIL import Image

img = Image.open("ill.png")
map = Image.open("map.jpg")
tub = Image.open("tub.jpg")
env = Image.open("env.png")
tree = Image.open("tree.png")

df = pd.read_csv("worldtemp.csv", encoding='cp1252')

# Calculate the yearly average temperatures
yearly_avg_temps = df.iloc[:, 2:].mean(axis=0)

# Store the average temperature values in a list
tempav = yearly_avg_temps.astype(float).values.tolist()
for i in range(len(tempav)):
    tempav[i] = round(tempav[i], 2)

# Generate the years range from 1961 to 2019
years = np.arange(1961, 2020, dtype=int)

X = years.reshape(-1, 1)
y = np.array(tempav).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
trend_line = reg.predict(X)

# Plot the line graph with trend line
plt.figure(figsize=(10, 6))
plt.plot(years, tempav, color='b', label='Climate Change')
plt.plot(years, trend_line, color='r', linestyle='--', label='Trend Line')
plt.xlabel('Year')
plt.ylabel('Climate Change')
plt.title('Yearly Average Temperature (1961-2019) with Trend Line')
plt.legend()
plt.grid(True)
graph = plt.gcf()


#sidebar
st.sidebar.image(img)
n = st.sidebar.text_input("**기온변화 예상**")
if n:
    if n.isdigit():
        climate = predict_temperature_change(int(n))
        st.sidebar.write(f"""
**{n}**년의 예상 기온 변화는 **{climate}°C** 입니다""")
    else:
        st.write("정수 값을 입력해주세요")

#main
tab1, tab2= st.tabs(['Graph' , 'Global Warming'])
with tab1:
    st.title("Climate Change Data Analysis")
    st.subheader("with Linear Regression")
    st.header('')
    st.pyplot(graph)

    st.write("""1961년부터 2021년까지의 세계 기후 변화를 분석한 그래프입니다""")
    for _ in range(5):
        st.write('')
    url = 'https://www.kaggle.com/datasets/sevgisarac/temperature-change'
    st.write("Source: [Kaggle : Temperature change](%s)" % url)

with tab2:
    st.title("지구온난화 문제점 & 개선방안")
    st.subheader("문제점")
    con, img = st.columns([3, 2])
    with con:
        st.write("""
**1. 해수면 상승**         
    기후 변화로 인해 얼음이 녹아내리고, 물의 부피가 커지며 해수면이 상승하고 있습니다
    이로 인해 해안 지역과 섬 국가에서 홍수, 침수 등의 문제가 발생하고 있으며, 
    저지대에 있는 국가들은 침수될 위험이 있습니다

                 
**2. 극단적 기후 변화**       
    지구 온난화로 인해 열대야 및 폭염 현상의 증가,
    잦은 태풍, 엘니뇨 및 라니냐 등 극단적인 기후 이벤트가 늘어나고 있습니다 
    
                 
**3. 농업 및 수산업 피해**           
    기후 변화로 인해 농작물의 생장 패턴 변화, 수확량 감소, 품질 저하 등의 문제를 유발합니다
    1980년대, 제주도에서 재배되던 한라봉은 2010년 김제에서 재배되고 있고, 
    보성에서 재배되던 녹차는 현재 고성에서 재배되고 있습니다
""")
        
        st.write('')
        st.write('')
        st.subheader("해결방안")
        st.write("""
**1. 재생 에너지 확대**
                 
태양광, 풍력, 지열 등의 재생 가능 에너지를 활용하여 화석 연료의 사용을 줄이는 것이 중요합니다
이를 통해 이산화탄소 등 온실가스의 배출을 줄일 수 있습니다

**2. 에너지 효율 향상**
                 
건물, 차량, 가전제품 등의 에너지 효율을 향상시키는 것도 중요한 해결방안입니다
LED 전구 사용 확대, 고효율 가전제품 선택, 에너지 효율적인 건물 디자인 등을 통해 에너지 사용량을 줄일 수 있습니다

**3. 숲 재생 및 보호**
                 
숲은 이산화탄소를 흡수하고 산소를 배출하는 등 지구 온난화를 완화하는 중요한 역할을 합니다
따라서 숲의 보호 및 재생에 힘써야 합니다

**4. 지속 가능한 농업 방법 활용**
                 
지속 가능한 농업 방법을 활용하면서도 농작물 생산을 유지하고, 비료 및 농약의 과도한 사용으로 인한 온실가스 배출을 줄일 수 있습니다.
""")
    with img:
        st.image(tub, width=250)
        st.write("▲가라앉고 있는 투발루")
        st.image(map, width=250)
        st.write("▲1980년대와 비교한 재배물 변동")
        for _ in range(5):
            st.write('')
        st.image(env, width=200)
        for _ in range(3):
            st.write('')
        st.image(tree, width=250)