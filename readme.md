## 진행 순서

1. 🖼️ 데이터 수집 및 전처리
- 텍스트 전처리 (토큰화 진행 -> 벡터화 진행)
    - tokenizing
    - vocabulary (단어장 만들기)
- 이미지 전처리 
    - resizing
    - nomalization

2. 🤖 모델 설계 및 학습
Encoder & Decoder 2단계로 구분
- Encoder -> 이미지를 받아서 주요 특징을 추출 (벡터화 진행)
- Decoder -> 추출한 이미지의 특징을 바탕으로 문장 생성

3. 📝 평가 및 최적화
모델이 생성한 캡션에 대한 평가 진행 BLEU와 같은 평가 지표를 활용하여 score 계산


## 테스트 데이터 확보
- Flickr8k 추가 사용(용량이 적어서 테스트로 돌리기 더 좋아보임)
    - https://www.kaggle.com/datasets/adityajn105/flickr8k
- 데이터가 부족한 경우 kaggle에서 제공하는 MS COCO dataset 추가 활용
    - (https://www.kaggle.com/datasets?search=MS+COCO)


