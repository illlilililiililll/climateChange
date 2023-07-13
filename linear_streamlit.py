import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from PIL import Image
import requests
from bs4 import BeautifulSoup

img = Image.open("ill.png")
map = Image.open("map.jpg")
tub = Image.open("tub.jpg")
env = Image.open("env.png")
tree = Image.open("tree.png")
earth = Image.open("earth.png")
scenario = Image.open("scenario.jpg")
g = Image.open("graph.jpg")
earthcase = Image.open("earthcase.jpg")
graph1 = Image.open("graph1.png")

#function
def predict_temperature_change(year):
    df = pd.read_csv("worldtemp.csv", encoding='cp1252')

    # Calculate the yearly average temperatures
    yearly_avg_temps = df.iloc[:, 2:].mean(axis=0)

    # Store the average temperature values in a list
    tempav = yearly_avg_temps.values.tolist()
    for i in range(len(tempav)):
        tempav[i] = round(tempav[i], 2)

    # Generate the years range from 1961 to 2019
    years = list(range(1961, 2020))

    # Perform linear regression
    X = np.array(years).reshape(-1, 1)
    y = np.array(tempav).reshape(-1, 1)
    reg = LinearRegression().fit(X, y)

    # Predict temperature change for the input year
    predicted_change = reg.predict(np.array(year).reshape(-1, 1))

    return round(predicted_change[0][0], 2)


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
#n = st.sidebar.text_input("**기온변화 예상**")
n = st.sidebar.slider("**기후변화 예상**", 2024, 2100)

climate = predict_temperature_change(int(n))
st.sidebar.write(f"""
**{n}**년엔 지구의 온도가 **{climate}°C** 상승합니다""")

#main
tab1, tab2, tab3, tab4, tab5= st.tabs(['Graph' , 'Global Warming', 'Warming Scenario', 'Problems & Solutions', 'News'])
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

with tab4:
    st.title("Problems & Solutions")
    for _ in range(2):
        st.write('')
    st.subheader("Problems")
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
        st.subheader("Solutions")
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

with tab2: #global warming
    cont, imge = st.columns([3, 2])
    with cont:
        st.title("Global Warming")
        st.write('')
        st.subheader("지구온난화란 ?")
        st.write("""####
**지구의 기온이 평균 이상으로 증가하는 현상** 
""")
        st.write('')
        st.write("""
2021년 기준 과거보다 **1.49°C**가량 증가했다

2021년 IPCC 6차 보고서에서는

이산화 탄소, 메테인, 질산, 할로젠 가스를 원인으로
규정하고 있다
""")
        for _ in range(2):
            st.write('')
        st.write("""
- 발생한 재난 사례
                 
    2010년 | 중부권 폭설 사태, 한반도 폭우, 퀸즈랜드 홍수
                 
    2011년 | 1월 한파, 중부권 폭우 사태, 캘리포니아 가뭄

    2015년 | 슈퍼 엘니뇨, 인도-파키스탄 폭염
                 
    2016년 | 두만강 유역 대홍수
                 
    2018년 | 일본 호우, 캘리포니아 산불
                 
    2019년 | 범지구적 이상 고온, 시베리아 산불
                 
    2020년 | 미국 서부 산불, 아시아 폭우 사태

                 
""")
    with imge:
        for _ in range(10):
            st.write('')
        st.image(earth)
        st.subheader('')
        st.image(g)
        st.write("ㅤㅤ▲지구 기온 변화 추이")

with tab3: #warming scenario
    st.title("Global Warming Scenario")
    st.subheader("지구 기후변화 시나리오")
    content, image = st.columns([3, 2])
    with content:
        for _ in range(3):
            st.write('')
        st.write('- **지구 온도가 1°C 상승하면?**')
        st.write('ㅤㅤ가뭄이 심각해짐')
        st.write('ㅤㅤ물 부족 인구 50,000,000명')
        st.write('ㅤㅤ10% 육상생물 멸종 위기')
        st.write('ㅤㅤ기후변화로 인해 300,000명 사망')
        st.write('ㅤㅤ킬리만자로의 만년빙 소멸')
        st.write('ㅤㅤ희귀 동식물 멸종')
        st.write('')
        st.write('- **지구 온도가 2°C 상승하면?**')
        st.write('ㅤㅤ사용 가능한 물 20% ~ 30% 감소')
        st.write('ㅤㅤ해빙으로 해수면 7m 상승')
        st.write('ㅤㅤ15% ~ 40% 북극생물 멸종 위기')
        st.write('ㅤㅤ말라리아 노출 최대 60,000,000명')
        st.write('ㅤㅤ이산화탄소의 흡수로 바다생물이 죽어감')
        st.write('ㅤㅤ저지대 도시 침수')
        st.write('- **지구 온도가 3°C 상승하면?**')
        st.write('ㅤㅤ기근으로 인한 사망 최대 3,000,000명')
        st.write('ㅤㅤ해안침수 피해 연 160,000,000명')
        st.write('ㅤㅤ20% ~ 50% 생물 멸종 위기')
        st.write('ㅤㅤ아마존 열대우림 파괴')
        st.write('ㅤㅤ허리케인으로 식량 생산 어려움')
        st.write('ㅤㅤ화재 발생')
        st.write('')
        st.write('- **지구 온도가 4°C 상승하면?**')
        st.write('ㅤㅤ사용 가능한 물 30 ~ 50% 감소')
        st.write('ㅤㅤ해안침수 피해 연 300,000,000명')
        st.write('ㅤㅤ아프리카 농산물 15% ~ 35% 감소')
        st.write('ㅤㅤ서남극 빙상 붕괴 위험')
        st.write('ㅤㅤ지중해 - 살인적인 폭염 및 가뭄')
        st.write('ㅤㅤ러시아 & 동유럽 - 눈이 내리지 않음')
        st.write('')
        st.write('- **지구 온도가 5°C 상승하면?**')
        st.write('ㅤㅤ군소도서국과 뉴욕, 런던 등 침수 위험')
        st.write('ㅤㅤ재난으로 인한 자본시장 붕괴')
        st.write('ㅤㅤ중국·인도 영향권 히말라야 빙하 소멸')
        st.write('ㅤㅤ핵무기가 동원된 전쟁 발발')
        st.write('ㅤㅤ거주 가능 지역에서 피난민 간 갈등 발생')
        st.write('')
        st.write('- **지구 온도가 6°C 상승하면?**')
        st.write('ㅤㅤ메탄하이드레이트 대량 분출로 인한 생물체 대멸종')
        st.write('')
        st.write('')
        st.write("-> 2018.10 IPCC : **지구온난화 1.5℃ 특별보고서** 채택")
        st.write("현재 추세에 따르면 2040년경 1.5℃ 상승")
        st.write("각별한 주의가 필요하다")

        for _ in range(5):
            st.write('')
        ipcc = 'https://www.ipcc.ch/sr15/'
        st.write("From: [IPCC : Global Warming of 1.5 ºC](%s)" % ipcc)
    with image:
        for _ in range(2):
            st.write('')
        st.image(scenario)
        for _ in range(12):
            st.write('')
        st.image(earthcase)
        st.write('지구의 기온별 상태')
        for _ in range(20):
            st.write('')
        st.image(graph1)

with tab5:
    def get_articles():
        url = "https://www.reuters.com/search/news?blob=global+warming&sortBy=date&dateRange=all"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        articles = soup.find_all('div', class_='search-result-content')
        articles_dict = {}

        for article in articles:
            title = article.find('h3', class_='search-result-title').text
            link = 'https://www.reuters.com' + article.find('a')['href']
            articles_dict[title] = link

        return articles_dict
    
    def main():
        st.title("지구온난화 관련 최신 기사")
        st.subheader('from [REUTERS](%s)'%'https://www.reuters.com')
        st.write('')
        st.write('')
        st.write('')
        articles = get_articles()
        for title, link in articles.items():
            st.write(f"[{title}]({link})")
            for _ in range(2):
                st.write('')
    main()