# ML Lecture 06
### Softmax Regression 기본 개념 + Softmax classifier 의 Cost 함수
#### Softmax Regression
> ![img](img/lec06-1.png)
> #### A, B, C 그룹이 있을때 어떻게 분류하지?

> ![img](img/lec06-2.png)
> A or Not, B or Not, C or Not 으로 분류해서 하면 되지 않나?

#### Hypothesis
![img](img/lec06-3.png)

> 행렬로 바꿔!
> ![img](img/lec06-5.png)

> 이걸 보통 이렇게 나타냄!
> ![img](img/lec06-6.png)

> 결과값!
> ![img](img/lec06-7.png)
> Softmax 에 넣으면 이렇게 나온다! --> `합이 1`
> ![img](img/lec06-9.png)

> #### 이걸 On Hot Encoding 을 하면 `(1, 0, 0)` 이런식으로 나타낼수 있다!

#### Cross-Entropy Cost Function
![img](img/lec06-10.png)
> `S` 는 예측값, `L`은 실제 분류값!

![img](img/lec06-11.png)
> 즉 정답을 맞추면 0의 Cost를 가지고, 정답을 맞추지 못하면 무한의 Cost를 가진다!