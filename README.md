# bbang-hyung-2
# 2강 분류
## 간단한 분류기의 예
### 분류기란?
강아지와 고양이의 사진을 보고 분유하는 기계

- 분류 : Classify
- 기계 : machine
- 분류기 : classifier
## 강아지와 고양이를 구분하는 분류기
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/9e0a486b-1ba5-4077-a5bf-cbac5967e157)
## 어떻게 선을 그려야 잘 구분할 수 있을까?
구분선과 강아지, 고양이 사이의 거리를 구하여 거리가 최대가 되는 선을 긋는다.

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/2ee4a109-77e4-45fa-a4cd-2fe1fa455106)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/5504e1e9-b970-4864-be14-16812bdd2e12)

## SVM 실습

### SVM 실습 1. Iris 데이터셋을 이용한 실습
- 데이터셋 로드

```python
from sklearn.datasets import load_iris
// 사이킷런 라이브러리에서 load_iris 패키지를 가져온다.

import pandas as pd
// 판다스 패키지를 가져온다.

iris = load_iris()
// 클래스 초기화 및 iris 객체 생성

df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
// iris 데이터에대한 데이터 프레임 생성

df['target'] = iris['target']
// 데이터 프레임의 타겟을 iris 타겟으로 변경

df.head()
// df 데이터 프레임에서 상위 5개 데이터 나타냄.

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/6cf86586-3e12-48f6-8035-e98864bebb33)

```

- 전처리
```python
from sklearn.model_selection import train_test_split
// 사이킷런 라이브러리에서 train_test_split 패키리를 가져온다.

x_train, x_val, y_train, y_val = train_test_split(df.drop(columns=['target']), df[['target']], test_size=0.2, random_state=2021)
// train_test_split으로 x에서는 타겟을 빼고 넣고 y에는 타겟만 넣고 Validation의 값은 20% train셋 크기는 80%로 랜덤으로 나눈다.

print(x_train.shape, y_train.shape)
// 몇개씩 나뉘었는지 shape를 사용하여 알아본다.

print(x_val.shape, y_val.shape)
// 몇개씩 나뉘었는지 shape를 사용하여 알아본다.
```
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/6c0c2504-3883-40e0-8bc1-267fbc2e6678)

- 모델정의
```python
from sklearn.svm import SVC
// 사이킷런 라이브러리에서 svc(support vector classifier) 패키지를 가져온다

model = SVC()
// svc에 대한 모델을 정의한다
```
- 학습 Training
```python
model.fit(x_train, y_train['target'])
// 모델에 x데이터와 y데이터를 넣어서 훈련시킨다.
// 정답을 알려주어서 훈련시키는 이유는 지도 학습이기 때문이다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/3fe778a9-f39b-4ac1-b61a-5dc4a6ff64b9)

- 검증 Validation
```python
from sklearn.metrics import accuracy_score
// 사이킷런 라이브러리에서 accuracy_score(정확도 점수) 패키지를 가져온다.

y_pred = model.predict(x_val)
// x_val 데이터를 예측한다.

accuracy_score(y_val, y_pred) * 100
// 예측한 데이터를 정답 값과 비교하여 정확도를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/9525f6c7-5f8d-4c1f-a523-3a5ee7a95ec3)

### SVM 실습 2. MANIST 데이터셋을 이용한 실습
손글씨 숫자 이미지를 보고 0-9를 분류하는 분류기

- 데이터셋 로드
```python
from sklearn.datasets import load_digits
// 사이킷런 라이브러리에서 load_digits 패키지를 가젼온다.

digits = load_digits()
// 클래스 초기화 및 digits 객체 생성

digits.keys()
//  digits 딕셔너리의 key 값을 나타낸다.
```
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/d2a45f4c-ad43-48b4-b6c0-bd548f436969)

- 데이터 시각화

```python
data = digits['data']
// data변수에 데이터를 저장한다.

data.shape
// data가 어떤 형태로 이루어져 있는지 확인
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/18b1572b-8d1a-433b-b106-10dfa8a07349)

```python
target = digits['target']
// target 변수에 타겟 데이터를 저장한다.

target.shape
// target가 어떤 형태로 이루어져 있는지 확인
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/64ebbe1f-8a71-4caf-92d2-94fc0f5a1e3e)

데이터는 1797개가 있다.

```python
target = digits['target']
// target 변수에 타겟 데이터를 저장한다.

target.shape
// target이 어떤 형태로 이루어져 있는지 확인
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/696e3425-8636-495f-9195-14fe5e990565)

타겟 값도 1797개가 있다. x값과 라벨값에 크기가 같다.
```python
import matplotlib.pyplot as plt
// matplotlib.pyplot 패키지를 가져온다

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
// 2행5열 16x8 사이즈에 차트를 만든다.

for i, ax in enumerate(axes.flatten()):
// flatten를 사용하여 축을 1차원으로 만든다.

  ax.imshow(data[i].reshape((8, 8)))
// 데이터에 i번째를 하나씩 다 뽑아오는데 reshape로 8x8 크기로 뽑아온다. 

  ax.set_title(target[i])
// 이 차트에 제목은 차겟에 i번째로 라벨을 지정한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/7ea78aaa-c9a4-4257-9f59-258d38bba0e9)

- 데이터 전처리 - 정규화
```python
data[0]
// 데이터에 0번째 인덱스를 뽑아온다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/5c627b4a-2eff-4262-a863-943cf921e48f)

```python
from sklearn.preprocessing import MinMaxScaler
// 사이킷런 패키지에서 MinMaxScaler 패키지를 가져온다

scaler = MinMaxScaler()
// 클래스 초기화 scaler 객체생성

scaled = scaler.fit_transform(data)
// data를 정규화시킨다.

scaled[0]
// scaled에 0번째 인덱스를 뽑아온다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/8ced4e65-56fe-412d-93c2-9a4e53a1b706)

- 데이터 전처리 - 데이터셋 분할
```python
from sklearn.model_selection import train_test_split
// 사이킷런 라이브러리에서 train_test_split 패키지를 가져온다.

x_train, x_val, y_train, y_val = train_test_split(scaled, target, test_size=0.2, random_state=2021)
// train_test_split를 사용하여 x에는 정규화된 데이터를 넣어주고 y에는 타겟을 넣어주고 val은 20%, train은 80%로 랜덤으로 나눈다.

print(x_train.shape, y_train.shape)
// 몇개씩 나뉘었는지 shape를 사용하여 알아본다.

print(x_val.shape, y_val.shape)
// 몇개씩 나뉘었는지 shape를 사용하여 알아본다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/f139166c-33b4-4613-8e0f-ff687b79869f)

- 학습, 검증
```python
from sklearn.svm import SVC
// 사이킷런 라이브러리에서 SVC 패키지를 가져온다.

from sklearn.metrics import accuracy_score
// 사이킷런 라이브러리에서 accuracy_score 패키지를 가져온다.

model = SVC()
// svc에 대한 모델을 정의한다.

model.fit(x_train, y_train)
// 모델을 훈련시킨다.

y_pred = model.predict(x_val)
// x_val 데이터 값을 예측한다.

accuracy_score(y_val, y_pred) * 100
// 예측한 값과 정답 값과 비교하여 정확도를 채점한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/0ac156ca-dfa7-4273-a1ed-6972d4d0b339)

- 검증 결과 시각화
```python
import matplotlib.pyplot as plt
// matplotlib.pyplot 패키지를 가져온다.

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
// 2행5열 16x8 사이즈에 차트를 만든다.

for i, ax in enumerate(axes.flatten()):
// flatten를 사용하여 축을 1차원으로 만든다.

  ax.imshow(x_val[i].reshape((8, 8)))
// 데이터에 i번째를 하나씩 다 뽑아오는데 reshape로 8x8 크기로 뽑아온다.

  ax.set_title((f'True: {y_val[i]}, Pred: {y_pred[i]}'))
// 각 그림의 이름은 ture : 정답 값, pred : 예측한 값으로 나타낸다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/a7bd5f84-2380-4d8a-b327-3859d09e5e55)

틀린게 있으면 모델을 튜닝하여 정확도를 100%로 만들어야한다.

## KNN 실습
K-최근접 알고리즘

KNN은 비슷한 특성을 가진 개체끼리 나누는 알고리즘이다. 예를 들어 하얀 고양이가 새로 나타났을 때 일정 거리안에 다른 개체들의 개수(k)를 보고 자신의 위치를 결정한다.

- k = 2 일때 고양이 분류

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/9ea556b0-f27e-4864-837b-6d1997a369a8)

## 샘플 데이터로 KNN 이해하기
k = 3

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
import random
// KNeighborsClassifier, make_classification, random 패키지를 가져온다.
// make_classification 패키지는 분류 문제에 대한 가상의 데이터셋을 생성한다.

x, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=1)
// n_samples: 생성할 데이터 포인트의 총 수
// n_features: 각 데이터 포인트가 가질 특성 또는 열의 수
// n_informative: 클래스와 관련된 유용한 정보를 가진 특성의 수
// n_redundant: 클래스와 관련이 없는 중복 특성의 수
// n_classes: 생성할 클래스의 수
// random_state: 데이터를 생성하기 위한 무작위 시드

red = x[y == 0]
blue = x[y == 1]
// x 데이터에서 y가 0이면 red, 1이면 blue가 나오게 한다.

new_input = [[random.uniform(-2, 2), random.uniform(-2, 2)]]
// -2부터 2 사이의 실수 중에서 랜덤 값을 리턴

plt.figure(figsize=(10, 10))
// 가로 10 세로 10짜리 차트를 만든다.

plt.scatter(x=red[:, 0], y=red[:, 1], c='r')
plt.scatter(x=blue[:, 0], y=blue[:, 1], c='b')
// red 데이터는 빨간색으로 나타내고 blue 데이터는 파란 데이터로 나타낸다.

model = KNeighborsClassifier(n_neighbors=3)
// 모델 정의

model.fit(x, y)
// 모델 훈련

pred = model.predict(new_input)
// 모델 값 예측

pred_label = 'red' if pred == 0 else 'blue'
// 모델 예측값이 0이면 red를 표기하고 아니면 blue를 표기한다.

plt.scatter(new_input[0][0], new_input[0][1], 100, 'g')
// new_input[0][0], new_input[0][1] 위치에 산점도 사이즈가 100인 초록색 점을 표기한다

plt.annotate(pred_label, xy=new_input[0], fontsize=16)
// plt.annotate : 주석 함수
// xy 위치에 pred_label 글자를 16 사이즈 폰트로 주석을 단다.

plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/a4b602cf-7322-4d95-8cff-ddfea3c7cbbe)

계속 초록색에 타겟이 변경된다. k-초근접 이웃으로 타겟이 무슨 색인지 예측하여 알 수 있다.
## MNIST 데이터셋을 이용한 실습
- 모델 정의, 학습

```python
from sklearn.neighbors import KNeighborsClassifier
# KNeighborsClassifier 패키지를 가져온다

model = KNeighborsClassifier(n_neighbors=5)
# KNeighborsClassifier에 대한 모델을 정의한다.

model.fit(x_train, y_train)
# 모델을 훈련시킨다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/312d179e-4cbb-44bc-9897-502b377e45a8)

- 검증

```python
y_pred = model.predict(x_val)
# x_val의 데이터 값을 예측한다.

accuracy_score(y_val, y_pred) * 100
# 예측한 값과 결과 값을 비교하여 정확도를 채점한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/89fcc8be-c14b-4b4e-8345-2fc0e8b53527)

- 최적의 k 검색

```python
for k in range(1, 11):
// 모델을 10번 학습시킨다.

  model = KNeighborsClassifier(n_neighbors=k)
// KNeighborsClassifier에 대한 모델을 정의한다.

  model.fit(x_train, y_train)
// 모델을 훈련시킨다.

  y_pred = model.predict(x_val)
// x_val의 데이터 값을 예측한다.

  print(f'k: {k}, accuracy: {accuracy_score(y_val, y_pred) * 100}')
// k:k, accuracy: 정답 값과 예측한 값을 비교했을 때 예측한 값에 정확도로 나타낸다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/2116425e-1a2f-4c76-9b15-073f10498073)
