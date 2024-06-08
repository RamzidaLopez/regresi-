import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Muat data dari CSV
data = pd.read_csv('D:/akbar/kuliah/semester 4/metode numerik/student_performance.csv')  # Ganti dengan path lengkap jika perlu

# Tampilkan lima baris pertama dari data
print(data.head())

# Pisahkan fitur dan target
X = data[['Hours Studied']]
y = data['Performance Index']

# Model Linear (Metode 1)
model_linear = LinearRegression()
model_linear.fit(X, y)

# Plot data dan hasil regresi linear
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, model_linear.predict(X), color='red', label='Regresi Linear')
plt.title('Linear Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.legend()

# Hitung galat RMS untuk Model Linear
rms_linear = mean_squared_error(y, model_linear.predict(X), squared=False)
print("Root Mean Squared Error (Linear):", rms_linear)

# Model Eksponensial (Metode 3)
X_log = np.log(X)
model_exponential = LinearRegression()
model_exponential.fit(X_log, y)

# Plot data dan hasil regresi eksponensial
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, model_exponential.predict(X_log), color='green', label='Regresi Eksponensial')
plt.title('Exponential Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.legend()

# Hitung galat RMS untuk Model Eksponensial
rms_exponential = mean_squared_error(y, model_exponential.predict(X_log), squared=False)
print("Root Mean Squared Error (Exponential):", rms_exponential)

plt.tight_layout()
plt.show()
