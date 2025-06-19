import numpy as np

# ---------- 1. 배열 생성 ----------
print("---------- 1. 배열 생성 ----------\n")

# 파이썬 리스트/튜플로부터 배열 생성
a = np.array([1, 2, 3])
print("파이썬 리스트/튜플로부터 배열 생성: %s \n" % (a))

# 모든 요소가 0/1/random인 배열 생성
shape = (5, 2)
a = np.zeros(shape) # np.ones(shape), np.empty(shape)
print("모든 요소가 0인 %s shape의 배열 생성: \n %s \n" % (shape, a))

# 지정된 값으로 채워진 배열 생성
shape = (10, )
value = 5
a = np.full(shape, value)
print("모든 요소가 %s인 %s shape의 배열 생성: \n %s \n" % (value, shape, a))

# i 부터 k steps 씩 j 에 제일 근접한 수 까지 배열 생성
i = 0
j = 10
k = 2
a = np.arange(i, j, k)
print("i=%s, j=%s, k=%s 인 배열 생성: \n %s \n" % (i, j, k, a))

# [i,j] 사이에 균등 간격으로 k개 요소 생성
i = 0
j = 10
k = 5
a = np.linspace(i,j,k)
print("i=%s, j=%s, k=%s 인 균등 간격으로 %s 개 요소 생성: \n %s \n" % (i, j, k, k, a))

# n x n 단위 행렬(identity matrix)
n = 3
a = np.eye(n)
print("n=%s 인 단위 행렬 생성: \n %s \n" % (n, a))

# ---------- 2. 배열 속성 ----------
print("---------- 2. 배열 속성 ----------\n")

# 배열의 각 차원 크기를 나타내는 튜플 (shape)
print("배열의 shape: %s \n" % (str(a.shape)))

# 배열의 차원 수 (ndim)
print("배열의 차원 수: %s \n" % (a.ndim))

# 배열의 총 요소 수 (size)
print("배열의 총 요소 수: %s \n" % (a.size))

# 배열의 각 요소의 데이터 타입 (dtype)
print("배열의 데이터 타입: %s \n" % (a.dtype))

# ----------- 3. 배열 인덱싱 ----------
print("---------- 3. 배열 인덱싱 ----------\n")

a = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
# 1D 배열: a[index], a[start:end], a[start:end:step]
# 2D 배열: a[row_index, col_index] 또는 a[row_slice, col_slice]

# 불리언 인덱싱
bool_index = a > 5
print("불리언 인덱싱: \n %s \n" % (a[bool_index]))

# 특정 행들 선택
rows = [0, 2]
print("특정 행들 선택: \n %s \n" % (a[rows, :]))

# 특정 열들 선택
cols = [0, 2]
print("특정 열들 선택: \n %s \n" % (a[:, cols]))

# 특정 위치의 요소들 선택
positions = np.array([[0, 1], [2, 2]])
print("특정 위치의 요소들 선택: \n %s \n" % (a[positions[:, 0], positions[:, 1]]))

# ----------- 4. 배열 형태 변경 ----------
print("---------- 4. 배열 형태 변경 ----------\n")

# 배열의 형태 변경 (reshape)
# 배열의 형태를 변경할 때는 총 요소 수가 동일해야 함; 예: 3x3 배열을 1x9 배열로 변경
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
new_shape = (1, 9)
reshaped_a = a.reshape(new_shape)
print("배열 형태 변경: \n %s \n" % (reshaped_a))

# resize는 배열의 크기를 변경하지만, 원본 배열을 변경하지 않음
# 배열의 크기를 변경할 때는 총 요소 수가 동일해야 하지 않음
new_shape = (8, 1)
resized_a = np.resize(a, new_shape)
print("배열 크기 조정: \n %s \n" % (resized_a))

# 배열을 1차원으로 평탄화 (flatten)
flattened_a = a.flatten()
print("배열 평탄화: \n %s \n" % (flattened_a))

# 전치 배열 (transpose)
transposed_a = a.T
print("배열 전치: \n %s \n" % (transposed_a))

# 여러 배열을 연결 (concatenate)
b = np.array([[10, 11, 12],
              [13, 14, 15],
              [16, 17, 18]])
concatenated_a = np.concatenate((a, b), axis=0)  # axis = 0 (행 방향), axis = 1 (열 방향)
print("여러 배열 연결: \n %s \n" % (concatenated_a))

# 배열 분할 (split)
n = 2  # 분할할 조각의 수
split_a = np.split(concatenated_a, n, axis=0)
print("배열 분할: \n %s \n" % (split_a))

# 요소 추가 (append)
# shape 이 맞아야 함
new_element = np.array([[19, 20, 21]])
appended_a = np.append(concatenated_a, new_element, axis=0)
print("요소 추가: \n %s \n" % (appended_a))

# 요소 삭제 (delete)
i = 0  # 삭제할 행의 인덱스
deleted_a = np.delete(appended_a, i, axis=0)  # 첫 번째 행 삭제
print("요소 삭제: \n %s \n" % (deleted_a))

# 요소 삽입 (insert)
# 삽입할 행의 shape 이 맞아야 함
i = 1  # 삽입할 위치의 인덱스
new_row = np.array([[22, 23, 24]])
inserted_a = np.insert(deleted_a, i, new_row, axis=0)
print("요소 삽입: \n %s \n" % (inserted_a))

# ---------- 5. 배열 연산 ----------
print("---------- 5. 배열 연산 ----------\n")

# 배열 간의 사칙연산 (+, -, *, /, %) element-wise opearations
# arr op arr / arr op scalar
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("배열 a: %s, 배열 b: %s" % (a, b))
print("배열 덧셈: %s \n" % (a + b))
print("배열 뺄셈: %s \n" % (a - b))
print("배열 곱셈: %s \n" % (a * b))
print("배열 나눗셈: %s \n" % (a / b))
print("배열 나머지: %s \n" % (a % b))

# 배열의 모든 요소 합
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
sum_a = np.sum(a)
print("배열의 모든 요소 합: %s \n" % (sum_a))

# 배열의 특정 축을 따라 합
sum_a_axis0 = np.sum(a, axis=0)  # 열 방향 합
sum_a_axis1 = np.sum(a, axis=1)  # 행 방향 합
print("배열의 열 방향 합: %s \n" % (sum_a_axis0))
print("배열의 행 방향 합: %s \n" % (sum_a_axis1))

# 배열의 평균
mean_a = np.mean(a)
print("배열의 평균: %s \n" % (mean_a))

# 배열의 특정 축을 따라 평균
mean_a_axis0 = np.mean(a, axis=0)  # 열 방향 평균
mean_a_axis1 = np.mean(a, axis=1)  # 행 방향 평균
print("배열의 열 방향 평균: %s \n" % (mean_a_axis0))
print("배열의 행 방향 평균: %s \n" % (mean_a_axis1))

# 배열의 중앙값
median_a = np.median(a)
print("배열의 중앙값: %s \n" % (median_a))

# 배열의 표준편차
std_a = np.std(a)
print("배열의 표준편차: %s \n" % (std_a))   

# 배열의 분산
var_a = np.var(a)
print("배열의 분산: %s \n" % (var_a))   

# 배열의 최대값, 최소값
max_a = np.max(a)
min_a = np.min(a)
print("배열의 최대값: %s \n" % (max_a))
print("배열의 최소값: %s \n" % (min_a))

# 배열의 최대값, 최소값의 인덱스
max_index = np.argmax(a)
min_index = np.argmin(a)
print("배열의 최대값 인덱스: %s \n" % (max_index))
print("배열의 최소값 인덱스: %s \n" % (min_index))

# 배열의 행렬곱 (dot product)   
dot_product = np.dot(a, a.T)
print("배열의 행렬곱: \n %s \n" % (dot_product))

# 행렬 곱 (matmul or @ operator)
matmul_product = np.matmul(a, a.T)
print("배열의 행렬 곱: \n %s \n" % (matmul_product))

# 배열의 제곱근 (element-wise square root)
sqrt_a = np.sqrt(a)
print("배열의 제곱근: \n %s \n" % (sqrt_a))

# 배열의 거듭제곱 (element-wise power)
power_a = np.power(a, 2)  # 각 요소를 제곱
print("배열의 거듭제곱: \n %s \n" % (power_a))

# 배열의 로그 (element-wise logarithm)
log_a = np.log(a)  # 자연 로그
print("배열의 로그: \n %s \n" % (log_a))

# 배열의 지수 (element-wise exponential)
exp_a = np.exp(a)  # 자연 지수
print("배열의 지수: \n %s \n" % (exp_a))

# 배열의 삼각 함수 (element-wise trigonometric functions)
sin_a = np.sin(a)  # 사인
cos_a = np.cos(a)  # 코사인
tan_a = np.tan(a)  # 탄젠트
print("배열의 사인: \n %s \n" % (sin_a))
print("배열의 코사인: \n %s \n" % (cos_a))
print("배열의 탄젠트: \n %s \n" % (tan_a))

# 배열의 역삼각 함수 (element-wise inverse trigonometric functions)
arcsin_a = np.arcsin(a)  # 역 사인
arccos_a = np.arccos(a)  # 역 코사인
arctan_a = np.arctan(a)  # 역 탄젠트
print("배열의 역 사인: \n %s \n" % (arcsin_a))
print("배열의 역 코사인: \n %s \n" % (arccos_a))
print("배열의 역 탄젠트: \n %s \n" % (arctan_a))

# 배열의 절댓값 (element-wise absolute value)
abs_a = np.abs(a)
print("배열의 절댓값: \n %s \n" % (abs_a))

# 배열의 부호 (element-wise sign)
sign_a = np.sign(a)  # 양수: 1, 음수: -1, 0: 0
print("배열의 부호: \n %s \n" % (sign_a))

# 배열의 정렬 (sort)
# 전체 배열 정렬
# axis=None: 전체 배열을 1차원으로 보고 정렬
# axis=0: 각 열을 기준으로 정렬
# axis=1: 각 행을 기준으로 정렬
a = np.array([[3, 1, 2],
              [6, 4, 5],
              [9, 7, 8]])
sorted_a = np.sort(a, axis=1)  # 전체 배열 정렬
print("배열 정렬: \n %s \n" % (sorted_a))

# 배열의 고유값 (unique values)
a = np.array([1, 2, 2, 3, 4, 4, 5])
unique_a = np.unique(a)
print("배열의 고유값: \n %s \n" % (unique_a))

# ---------- 6. 브로드캐스팅 (broadcasting) ----------
print("---------- 6. 브로드캐스팅 (broadcasting) ----------\n")

# 브로드캐스팅은 서로 다른 shape의 배열 간 연산을 가능하게 함
# 작은 배열이 큰 배열의 shape에 맞춰 자동으로 확장됨
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])  # 1D 배열
print("배열 a: \n %s \n" % (a))
print("배열 b: \n %s \n" % (b))
# 브로드캐스팅을 통해 a와 b의 덧셈이 가능
broadcasted_sum = a + b  # b가 a의 각 행에 더해짐
print("브로드캐스팅 덧셈 결과: \n %s \n" % (broadcasted_sum))

# 브로드캐스팅을 이용한 곱셈
c = np.array([[1, 2, 3]])  # 3x1 배열
print("배열 c: \n %s \n" % (c))
# a와 c의 곱셈도 브로드캐스팅을 통해 가능
broadcasted_product = a * c  # c가 a의 각 열에 곱해짐
print("브로드캐스팅 곱셈 결과: \n %s \n" % (broadcasted_product))

# 브로드캐스팅을 이용한 조건부 연산
d = np.array([[1, 2, 3],
              [4, 5, 6]])
condition = d > 3  # 불리언 배열 생성
print("배열 d: \n %s \n" % (d))
print("조건 d > 3: \n %s \n" % (condition))
# 조건을 만족하는 요소에만 10을 더함
result = np.where(condition, d + 10, d)  # 조건을 만족하면 d + 10, 아니면 d
print("조건부 연산 결과: \n %s \n" % (result))

# ---------- 7. 랜덤 함수 ----------
print("---------- 7. 랜덤 함수 ----------\n")

# seed 설정
# 난수 생성의 재현성을 위해 seed를 설정
np.random.seed(42)

# 난수 생성
# 0과 1 사이의 균등 분포 난수 생성
random_uniform = np.random.rand(3, 2)  # 3x2 배열
print("0과 1 사이의 균등 분포 난수 생성: \n %s \n" % (random_uniform))

# 정규 분포 난수 생성
random_normal = np.random.randn(3, 2)  # 평균 0, 표준편차 1인 정규 분포
print("정규 분포 난수 생성: \n %s \n" % (random_normal))

# 특정 범위의 정수 난수 생성
low = 1
high = 10
size = (3, 2)
random_integers = np.random.randint(low, high, size)  # low 이상 high 미만의 정수 난수
print("특정 범위의 정수 난수 생성: \n %s \n" % (random_integers))

# ---------- 8. 파일 입출력 ----------
print("---------- 8. 파일 입출력 ----------\n")

# 배열을 텍스트 파일에서 읽기
# np.loadtxt(filename)

# 배열을 텍스트 파일에 쓰기
# np.savetxt(filename, array)

# 배열을 바이너리 파일에서 읽기
# np.load(filename)

# 배열을 바이너리 파일에 쓰기
# np.save(filename, array) 

# 배열을 압축된 바이너리 파일에 쓰기
# np.savez_compressed(filename, array)

# 배열을 CSV 파일에서 읽기
# np.genfromtxt(filename, delimiter=',')

# 배열을 CSV 파일에 쓰기
# np.savetxt(filename, array, delimiter=',')

# 배열을 JSON 파일에서 읽기
# import json
# with open(filename, 'r') as f:
#     array = np.array(json.load(f))

# 배열을 JSON 파일에 쓰기
# import json
# with open(filename, 'w') as f:
#     json.dump(array.tolist(), f)