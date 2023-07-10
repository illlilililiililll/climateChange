import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("worldtemp.csv", encoding='cp1252')

# Calculate the yearly average temperatures
yearly_avg_temps = df.iloc[:, 2:].mean(axis=0)

# Store the average temperature values in a list
tempav = yearly_avg_temps.values.tolist()
for i in range(len(tempav)):
    tempav[i] = round(tempav[i], 2)

# Generate the years range from 1961 to 2019
years = list(range(1961, 2020))

X = np.array(years).reshape(-1, 1)
y = np.array(tempav).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
trend_line = reg.predict(X)

# Plot the line graph with trend line
plt.figure(figsize=(10, 6))
plt.plot(years, tempav, color='b', label='Average Temperature')
plt.plot(years, trend_line, color='r', linestyle='--', label='Trend Line')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.title('Yearly Average Temperature (1961-2019) with Trend Line')
plt.legend()
plt.grid(True)
plt.show()