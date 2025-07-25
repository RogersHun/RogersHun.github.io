---
title: 추천시스템의 평가지표
description: Precision, Recall, NDCG, Hit Rate, MAE, RMSE, MAPE, Negative Sampling, Leave-One-Out
date: 2025-07-04 11:33:00 +0800
categories: [AI, Recommender System]
pin: true
math: true
mermaid: true
---

> ### **Intro**
추천 시스템 모델을 고안하였다면, 이 모델이 얼마나 좋은지에 대한 평가 지표를 필수적으로 다루어야 한다.   
따라서, 해당 포스팅에서는 추천 시스템의 평가 지표에 대한 부분을 정리하고자 한다.

>#### **명시적 피드백(Explicit Feedback) vs 암묵적 피드백(Implicit Feedback)**
추천 시스템의 거의 모든 모델은 사용자의 피드백을 통해 학습 및 모델링이 진행된다. 이러한 피드백은 크게 두가지로 분류된다.   
- **명시적 피드백**   
**사용자가 명시적(직접적)으로 자신의 선호를 표현한 정보**이다. 평점(1~5점), 좋아요, 리뷰 등이 이에 속하는데 이러한 명시적 데이터에는 사용자의 아이템에 대한 긍정, 부정 선호도가 명확히 드러나 모델을 학습시키기 비교적 수월하다.   
<br>
- **암묵적 피드백**  
직접적으로 선호가 표현되진 않지만, **행동을 통해 암묵적(간접적)으로 선호를 추론할 수 있는 피드백 정보**이다. 아이템을 클릭 및 조회한 기록, 페이지 체류 시간, 구매 기록 등이 이에 속한다. 
<br>

>#### **Negative Sampling**
위에서 언급한 암묵적 피드백에는 오직 긍정(Positive) 신호만 존재한다는 딜레마가 존재한다.<br>
예를 들어, 사용자가 아이템A에 대해서 클릭을 한 정보가 있다면 사용자가 해당 아이템에 선호가 존재한다고 생각할 수 있다. 그러나, 다른 아이템B에 대해서 클릭하지 않았다고 해서 그 상품을 싫어한다던가, 그 상품의 존재를 몰랐다라고 단정지을 순 없다. 즉, "클릭하지 않음" = "싫음(Negative)"를 의미하지는 않다는 것이다.
<br>
이러한 관점에서 모델에게 "정답(Positive)" 샘플만 계속 보여주게 된다면, 모델은 사용자가 무엇을 좋아하는지는 알게 될지 몰라도, 무엇을 좋아하지 않는지는 전혀 알지 못하게 된다. 결국, 모든 아이템을 좋아한다고 판단하는 모델이 되게 될 수 있다.
<br>
따라서, 이를 개선하기 위해 모델에 "오답"도 함께 가르쳐야 한다. 이러한 방식이 Negative Sampling인데, 상호작용이 없었던 수많은 아이템 중에서 "이건 사용자가 좋아하지 않을 거야"라고 가정하는 가짜 "부정(Negative)" 샘플을 인위적으로 만들어주는 과정이다.   
<br>
내용 추가 예정..
