import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

train_df = pd.read_csv('Kaggle/train.csv')
# test_df = pd.read_csv('Kaggle/test.csv')

# Preprocess the training data
# Drop unnecessary columns
train_df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'], inplace=True)

# Tokenization of 'Name' column
# train_df['Name'] = train_df['Name'].apply(lambda x : " ".join([v.strip(",()[].\"'") for v in x.split(" ")]))

# Categorical encoding
train_df['Sex'] = train_df['Sex'].astype('category').cat.codes

# Categorical encoding for 'Embarked' (one-hot encoding)
encoded_embarked = pd.get_dummies(train_df['Embarked'], prefix='Embarked')
for col in ['Embarked_C', 'Embarked_Q', 'Embarked_S']:
    if col in encoded_embarked.columns and encoded_embarked[col].dtype == 'bool':
        encoded_embarked[col] = encoded_embarked[col].astype(int)

train_df = pd.concat([train_df, encoded_embarked], axis=1)
train_df.drop(columns=['Embarked'], inplace=True)

# Fill missing values

# Fill missing 'Age' with the median
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

y = train_df['Survived']
X = train_df.drop(columns=['Survived'])

# Convert DataFrame to NumPy arrays
X_train = X.values
y_train = y.values

# define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # 이진 분류
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 훈련 손실과 검증 손실 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 훈련 정확도와 검증 정확도 시각화
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()