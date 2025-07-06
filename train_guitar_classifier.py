import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2 # 사전 훈련된 모델
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. 환경 설정 및 데이터셋 경로 정의 ---
dataset_dir = 'downloaded_guitars'

# 이미지 크기 및 배치 크기 정의
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # 한 번에 처리할 이미지 수 (GPU 메모리 및 데이터셋 크기에 따라 조절)
NUM_CLASSES = 2 # 일렉트릭 기타, 통기타 (2개 클래스)
EPOCHS = 20 # 훈련 반복 횟수 (데이터셋 크기, 모델 성능에 따라 조절)

# --- 2. 데이터 로드 및 전처리 (ImageDataGenerator 사용) ---
# Keras의 ImageDataGenerator를 사용하여 데이터 증강 및 전처리
# validation_split을 사용하여 자동으로 훈련/검증 세트 분할
train_datagen = ImageDataGenerator(
    rescale=1./255,                 # 픽셀 값을 0-1 사이로 정규화
    rotation_range=20,              # 0~20도 범위에서 무작위 회전
    width_shift_range=0.2,          # 가로 방향으로 0~20% 무작위 이동
    height_shift_range=0.2,         # 세로 방향으로 0~20% 무작위 이동
    shear_range=0.2,                # 전단 변환 (찌그러뜨리기)
    zoom_range=0.2,                 # 0~20% 무작위 줌
    horizontal_flip=True,           # 좌우 반전
    fill_mode='nearest',            # 빈 공간을 주변 픽셀로 채움
    validation_split=0.2            # 전체 데이터의 20%를 검증 세트로 사용
)

# 검증 데이터는 증강을 적용하지 않고 정규화만 수행 (모델 평가의 일관성을 위해)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 훈련 데이터셋 로드
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # 2개 이상의 클래스 분류 (원-핫 인코딩)
    subset='training'         # 훈련 세트로 사용
)

# # 검증 데이터셋 로드
validation_generator = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # 2개 이상의 클래스 분류
    subset='validation'       # 검증 세트로 사용
)

# 클래스 이름 확인 (인덱스와 실제 클래스 이름 매핑)
class_names = list(train_generator.class_indices.keys())
print(f"클래스 이름: {class_names}") # ['acoustic', 'electric']

# --- 3. 모델 구축 (전이 학습 - MobileNetV2 사용) ---

# MobileNetV2 모델 불러오기 (ImageNet으로 사전 훈련된 가중치 사용)
# include_top=False: MobileNetV2의 최상위 분류 계층(1000개 클래스)을 제외
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# 불러온 모델의 가중치를 고정 (사전 훈련된 특징 추출 부분은 학습하지 않음)
for layer in base_model.layers:
    layer.trainable = False

# 새로운 분류 계층 추가
x = base_model.output
x = GlobalAveragePooling2D()(x) # 특징 맵을 단일 벡터로 평탄화
x = Dense(128, activation='relu')(x) # 추가적인 Dense 레이어 (선택 사항)
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # 최종 분류 레이어 (2개 클래스, softmax 활성화)

# 최종 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.0001), # 낮은 학습률 사용 (전이 학습에 적합)
              loss='categorical_crossentropy',       # 다중 클래스 분류 손실 함수
              metrics=['accuracy'])                  # 정확도를 평가 지표로 사용

model.summary() # 모델 구조 요약 출력

# --- 4. 모델 훈련 ---
print("\n--- 모델 훈련 시작 ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE, # 한 에포크당 스텝 수
    validation_steps=validation_generator.samples // BATCH_SIZE # 한 에포크당 검증 스텝 수
)
print("--- 모델 훈련 완료 ---")

# --- 5. 모델 평가 ---
print("\n--- 모델 최종 평가 ---")
loss, accuracy = model.evaluate(validation_generator)
print(f"검증 데이터셋 손실(Loss): {loss:.4f}")
print(f"검증 데이터셋 정확도(Accuracy): {accuracy:.4f}")

# --- 6. 훈련 과정 시각화 ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# --- 7. 혼동 행렬 (Confusion Matrix) 및 분류 보고서 생성 (옵션) ---
# 실제 데이터의 예측 결과를 확인하여 모델의 성능을 더 자세히 분석합니다.
print("\n--- 혼동 행렬 및 분류 보고서 생성 ---")
validation_generator.reset() # 제너레이터 초기화

# 검증 데이터셋의 모든 배치에 대해 예측 수행
y_pred_probs = model.predict(validation_generator)
y_pred = np.argmax(y_pred_probs, axis=1) # 가장 높은 확률을 가진 클래스 인덱스 선택

# 실제 라벨 가져오기
y_true = validation_generator.classes[validation_generator.index_array] # 검증 데이터의 원본 라벨
# 중요한 점: flow_from_directory는 기본적으로 알파벳 순서로 클래스 인덱스를 부여합니다.
# class_names 변수를 통해 실제 라벨과 매핑되는지 확인해야 합니다.
# 예: {'acoustic': 0, 'electric': 1} 또는 그 반대

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("\n--- 분류 보고서 ---")
print(classification_report(y_true, y_pred, target_names=class_names))


# --- 8. 모델 저장 ---
model_save_path = 'guitar_classifier_model.h5'
model.save(model_save_path)
print(f"\n모델이 '{model_save_path}' 경로에 성공적으로 저장되었습니다.")