#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt


file_path = r"Lab Session1 Data (1).xlsx"
sheet_name = 'IRCTC Stock Price'
data = pd.read_excel(file_path, sheet_name=sheet_name)



price_data = data['Price']
mean_price = statistics.mean(price_data)
variance_price = statistics.variance(price_data)
print("Mean of Price data:", mean_price)

print("Variance of Price data:", variance_price)

data['Date'] = pd.to_datetime(data['Date'])
wednesday_data = data[data['Date'].dt.day_name() == 'Wednesday']
sample_mean_wednesday = statistics.mean(wednesday_data['Price'])
print("Sample mean on Wednesdays:", sample_mean_wednesday)

print("Population mean (overall mean):", mean_price)

april_data = data[data['Date'].dt.month == 4]
sample_mean_april = statistics.mean(april_data['Price'])
print("Sample mean in April:", sample_mean_april)


loss_probability = len(data[data['Chg%'] < 0]) / len(data)
print("Probability of making a loss over the stock:", loss_probability)


profit_on_wednesday_probability = len(wednesday_data[wednesday_data['Chg%'] > 0]) / len(wednesday_data)
print("Probability of making a profit on Wednesday:", profit_on_wednesday_probability)


conditional_profit_probability = len(wednesday_data[wednesday_data['Chg%'] > 0]) / len(wednesday_data)
print("Conditional probability of making profit on Wednesday:", conditional_profit_probability)


plt.scatter(data['Date'].dt.weekday, data['Chg%'])
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Chg% Data vs. Day of the Week")
plt.xticks(np.arange(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()


# In[ ]:




