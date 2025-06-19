import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------- 1. 모델 구축 (Model Building - Keras Sequential API) ----------
print("---------- 1. 모델 구축 (Keras Sequential API) ----------\n")

# Sequential 모델: 레이어를 순서대로 쌓아 올립니다.
model = keras.Sequential([
    # 첫 번째 Dense (완전 연결) 레이어
    # units: 뉴런(출력 차원)의 수
    # activation: 활성화 함수 (relu, sigmoid, softmax, tanh 등)
    # input_shape: (첫 번째 레이어에만 필요) 입력 데이터의 형태 (배치 크기 제외)
    layers.Dense(units=64, activation='relu', input_shape=(10,)), # 10개의 특성을 가진 입력에 대해 64개의 뉴런

    # 두 번째 Hidden Dense 레이어
    layers.Dense(units=32, activation='relu'),

    # 출력 레이어 (예: 이진 분류 - 0 또는 1)
    layers.Dense(units=1, activation='sigmoid') # 이진 분류의 경우 1개의 뉴런과 sigmoid 활성화
])

# 모델 요약 정보 확인
print("모델 요약:\n")
model.summary()
print("\n")

# 다중 클래스 분류 출력 레이어 예시 (예: 10개 클래스)
# model_multiclass = keras.Sequential([
#     layers.Dense(units=64, activation='relu', input_shape=(10,)),
#     layers.Dense(units=10, activation='softmax') # 10개 클래스 분류를 위한 10개 뉴런과 softmax 활성화
# ])


# ---------- 2. 모델 컴파일 (Model Compilation) ----------
print("---------- 2. 모델 컴파일 ----------\n")

# model.compile(optimizer, loss, metrics)
# optimizer: 가중치를 업데이트하는 알고리즘 (예: 'adam', 'sgd', 'rmsprop')
# loss: 손실 함수 (모델이 얼마나 잘 예측하는지 측정, 최소화할 대상)
#   - 'binary_crossentropy': 이진 분류
#   - 'categorical_crossentropy': 원-핫 인코딩된 다중 클래스 분류
#   - 'sparse_categorical_crossentropy': 정수 레이블 다중 클래스 분류
#   - 'mse' (Mean Squared Error): 회귀
# metrics: 훈련과 평가를 모니터링할 지표 (예: 'accuracy', 'mse', 'mae')

# Adam 옵티마이저를 기본 학습률로 사용 (가장 일반적)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("모델이 Adam 옵티마이저로 컴파일되었습니다.\n")

# 특정 학습률을 가진 Adam 옵티마이저 인스턴스 생성
# custom_adam = keras.optimizers.Adam(learning_rate=0.001)
# model.compile(optimizer=custom_adam,
#               loss='mse', # 회귀 문제의 경우
#               metrics=['mae']) # 평균 절대 오차


# ---------- 3. 모델 훈련 (Model Training) ----------
print("---------- 3. 모델 훈련 ----------\n")

# 더미 데이터 생성 (실제 데이터 대신 예시로 사용)
# X: 입력 특성, y: 정답 레이블
X_train = np.random.rand(100, 10).astype(np.float32) # 100개 샘플, 10개 특성
y_train = np.random.randint(0, 2, size=(100, 1)).astype(np.float32) # 100개 샘플, 이진 레이블

# model.fit(x=입력 데이터, y=정답 데이터, epochs=훈련 반복 횟수, batch_size=배치 크기, validation_data=검증 데이터)
print("모델 훈련 시작 (예시 더미 데이터):\n")
history = model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0) # verbose=0: 훈련 과정 출력 안 함
print("모델 훈련 완료. (5 에포크, 배치 크기 32)\n")

# 훈련 이력 (history) 확인
# print("훈련 손실 (마지막 에포크):", history.history['loss'][-1])
# print("훈련 정확도 (마지막 에포크):", history.history['accuracy'][-1])


# ---------- 4. 모델 평가 및 예측 (Model Evaluation & Prediction) ----------
print("---------- 4. 모델 평가 및 예측 ----------\n")

# 더미 테스트 데이터 생성
X_test = np.random.rand(20, 10).astype(np.float32)
y_test = np.random.randint(0, 2, size=(20, 1)).astype(np.float32)

# 모델 평가
# model.evaluate(x=입력 데이터, y=정답 데이터, batch_size=배치 크기)
# 반환 값: [손실, 메트릭1, 메트릭2, ...]
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"테스트 손실: {loss:.4f}")
print(f"테스트 정확도: {accuracy:.4f}\n")

# 예측
# model.predict(x=입력 데이터, batch_size=배치 크기)
predictions = model.predict(X_test[:5], verbose=0) # 첫 5개 샘플에 대해 예측
print("첫 5개 샘플에 대한 예측 (확률):\n", predictions.flatten())
# 이진 분류의 경우, 확률을 이진 값으로 변환 (예: 0.5 이상이면 1)
predicted_classes = (predictions > 0.5).astype(int).flatten()
print("첫 5개 샘플에 대한 예측 (클래스):\n", predicted_classes)
print("첫 5개 샘플의 실제 레이블:\n", y_test[:5].flatten())
print("\n")


# ---------- 5. 모델 저장 및 로드 (Model Saving & Loading) ----------
print("---------- 5. 모델 저장 및 로드 ----------\n")

# 모델 저장 (SavedModel 형식)
model_path = 'my_first_tf_model'
model.save(model_path)
print(f"모델이 '{model_path}' 경로에 저장되었습니다.\n")

# 모델 로드
loaded_model = keras.models.load_model(model_path)
print(f"모델이 '{model_path}' 경로에서 성공적으로 로드되었습니다.\n")

# 로드된 모델 평가 (원래 모델과 동일한 결과여야 함)
loaded_loss, loaded_accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)
print(f"로드된 모델의 테스트 손실: {loaded_loss:.4f}")
print(f"로드된 모델의 테스트 정확도: {loaded_accuracy:.4f}\n")


# ---------- 6. 텐서 기본 조작 (Basic Tensor Operations) ----------
print("---------- 6. 텐서 기본 조작 ----------\n")

# 텐서 생성
tensor_a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
tensor_b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
print("텐서 A:\n", tensor_a.numpy()) # .numpy()로 NumPy 배열로 변환하여 출력
print("텐서 B:\n", tensor_b.numpy())

# 요소별 덧셈
add_result = tensor_a + tensor_b
print("\n텐서 덧셈 (요소별):\n", add_result.numpy())

# 행렬 곱셈
matmul_result = tf.matmul(tensor_a, tensor_b) # 또는 tensor_a @ tensor_b
print("텐서 행렬 곱셈:\n", matmul_result.numpy())

# 합계 (모든 요소의 합)
sum_tensor = tf.reduce_sum(tensor_a)
print("\n텐서 요소의 합:", sum_tensor.numpy())

# 특정 축(axis)을 따라 합계
sum_axis0 = tf.reduce_sum(tensor_a, axis=0) # 열 방향 합
sum_axis1 = tf.reduce_sum(tensor_a, axis=1) # 행 방향 합
print("텐서 열 방향 합:", sum_axis0.numpy())
print("텐서 행 방향 합:", sum_axis1.numpy())

# 평균
mean_tensor = tf.reduce_mean(tensor_a)
print("텐서 요소의 평균:", mean_tensor.numpy())

# 데이터 타입 변환
casted_tensor = tf.cast(tensor_a, dtype=tf.int32)
print("데이터 타입 변환 (float32 -> int32):\n", casted_tensor.numpy())