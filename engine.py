import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. GENERATE AUSTRALIAN BANKING DATA
def generate_au_data():
    # Normal customers (Transactions < $10k)
    normal = np.random.normal(loc=[500, 10], scale=[100, 2], size=(1000, 2))
    # "Zero-Day" Fraud (Structuring and Synthetic ID behavior)
    fraud = np.array([[9950, 8], [15000, 1], [450, 1]]) 
    return normal, fraud

data_normal, data_fraud = generate_au_data()

# 2. SCALE DATA (0 to 1)
data_min, data_max = data_normal.min(axis=0), data_normal.max(axis=0)
norm_train = (data_normal - data_min) / (data_max - data_min)
norm_test = (data_fraud - data_min) / (data_max - data_min)

# 3. BUILD THE VAE-STYLE ENGINE
model = models.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='relu'), # Latent Space Bottleneck
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='sigmoid')
])
model.compile(optimizer='adam', loss='mse')

# 4. TRAIN & DETECT
print("Training Anomaly Engine...")
model.fit(norm_train, norm_train, epochs=30, verbose=0)
predictions = model.predict(norm_test)
mse = np.mean(np.square(norm_test - predictions), axis=1)

print("\n--- AUSTRAC ALERT LOG ---")
for i, score in enumerate(mse):
    status = "⚠️ HIGH RISK" if score > 0.02 else "✅ LOW RISK"
    print(f"App {i+1}: Error {score:.4f} -> {status}")

# 5. VISUAL PROOF (This will open a window)
plt.scatter(norm_train[:,0], norm_train[:,1], alpha=0.3, label='Normal')
plt.scatter(norm_test[:,0], norm_test[:,1], color='red', label='Fraud')
plt.legend()
plt.title("FinCrime Detection Map")
plt.show()

model.save('fincrime_vae_model.keras')