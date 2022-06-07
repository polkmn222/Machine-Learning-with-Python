# Machine Learning with Python    
개인공부      
실행프로그램 : Jupyter notebook(Anaconda3)    
교제 :  Introduction to Machine Learning with Python    
	파이썬 라이브러리를 활용한 머신러닝 번역개정2판    
    
### chap01 소개   
* Numpy, SciPy, matplotlib, pandas, mglearn    
* version     
* iris

### chap02-1 지도학습
* load_breast_cancer
* load_boston
* make_blobs
* import train_test_split
* import KNeighborsRegressor
* import LinearRegression
* import Lasso
* import LinearSVC

### chap02-2 지도학습
* 다중클래스 분류형 선형모델
* 나이브 베이즈 분류기
* 그레디언트 부스팅 회귀트리

### chap02-3 지도학습
* 배깅(Bagging)
* 에이다부스트(AdaBoost)
* 엑스트라 트리(Extra trees)
* 히스토그램 기반 그레이디언트 부스팅(
Histogram-based Gradient Boostring)

### chap02-4 지도학습
* 커널 서포트 벡터 머신(kernelized support vector machines)
* SVM 이해하기
* SVM 매개변수 튜닝
* SVM을 위한 데이터 전처리
* 신경망(딥러닝)
* 신경망 튜닝
* 분류 예측의 불확실성 추정
* 결정 함수
* 다중 분류에서의 불확실성

### chap03-1 비지도 학습과 데이터 전처리
* Quantile Transformer & PowerTransformer
* 지도 학습에서 데이터 전처리 효과
* 차원 축소, 특성 추출, 매니폴드 학습
* 주성분 분석(PCA)

### chap03-2 비지도 학습과 데이터 전처리
* 고유얼굴(eigenface) 특성 추출
* 비음수 행렬분해(NMF)

### chap03-3 비지도 학습과 데이터 전처리
* t-SNE를 이용한 매니폴드 학습
* 군집(Clustering)
* k-평균 군집(k-means)
* 병합 군집(agglomerative clustering)
* DBSCAN(density-based spatial clustering of applications with noise)

### chap04 데이터 표현과 특성 공학
* 범주형 변수
  원-핫-인코딩(one_hot_encoding) or 가변수(dummy variable), 원-아웃-오브-엔 인코딩(one-out-of-N encoding)   
* 숫자로 표현된 범주형 특성
* OneHotEncoder와 ColumnTransfomer : scikit-learn으로 범주형 변수 다루기
* make_column_transformer로 간편하게 ColumnTransformer 만들기
* 구간 분할, 이산화 그리고 선형 모델, 트리모델
* 상호작용과 다항식
* 일변량 비선형 변환
* 특성 자동 선택
1) 일변량 통계(univariage statistics)   
2) 모델 기반 선택(model-based selection)   
3) 반복적 선택(iterative selection)   
* 전문가 지식 활용

### chap05-1 모델 평가와 성능 향상
* 교차검증(Cross-validation)
1) scikit-learn의 교차 검증
* 계층별 k-겹 교차 검증과 그외 전략들
* 교차 검증 상세 옵션
* LOOCV(Leave-one-out croses-validation)
* 임의 분할 교차 검증(shuffle-split cross-validation)
* 그룹별 교차 검증

### chap05-2 모델 평가와 성능 향상
* 반복 교차 검증
* 그리드 서치(grid search)
* 교차 검증을 사용한 그리드 서치
* 교차 검증 결과 분석
* 비대칭 매개변수 그리드 탐색
* 중첩 교차 검증(nested cross-validation)
* 교차 검증과 그리드 서치 병렬화

### chap05-3 모델 평가와 성능 향상
* 불균형 데이터셋
* 오차행렬(confusion matrix)
* 불확실성 고려
* 정밀도-재현율 곡선과 ROC 곡선
* ROC와 AUC
* 다중 분류의 평가 지표
* 회기의 평가 지표

### chap06 알고리즘 체인과 파이프라인
* 데이터 전처리와 매개변수 선택
* 파이프라인 구축하기
* 그리드 서치에 파이프라인 적용하기
* 정보 누설에 대한 예시
* 파이프라인 인터페이스
make_pipeline을 사용한 파이프라인 생성    
1) 단계 속성에 접근하기   
2) 그리드 서치 안의 파이프라인 속성에 접근하기    
* 전처리와 모델의 매개변수를 위한 그리드 서치
중복계산 피하기

### chap07-1 텍스트 데이터 다루기
* 문자열 데이터 타입
* 예제 애플리케이션 : 영화 리뷰 감성 분석
* 텍스트 데이터를 BOW(bag of words)로 표현하기
1) 샘플데이터에 BOW적용하기    
2) 영화 리뷰에 대한 BOW    
* 불용어(stopword)
* tf-idf로 데이터 스케일 변경하기
* 모델 계수 조사
* 여러 단어로 만든 BOW (n-그램)
* 고급 토큰화, 어간 추출, 표제어 추출
* 토픽 모델링과 문서 군집화
1) LDA


### chap07-2
* KoNLPy를 사용한 영화 리뷰 분석
* Okt
* Mecab

### chap08 마무리
* 나만의 추정기 만들기

    
  
        
   


    

   
    
   