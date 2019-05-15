# HTML

### HTML
`HyperText Markup Language`

### 기본 문법

시작태그와 닫히는 태그가 있음.

#### 주석 : ```<!-- 주석내용 -->```
```html
<!-- 안녕 나는 주석이야 -->
```

#### 제목 : ```<h1> </h1>```
```html
<h1>START!</h1>
<!-- 숫자가 바뀌면 크기가 바뀜 -->
```

#### 강조하기 : ```<strong> </strong>```
```html
Hello This is my First <strong>HTML Programming</strong>.
```

#### 하이퍼텍스트
> ### 속성?
> 태그에 기능을 추가한다고 생각하면 됨.\
> ```<a>``` 태그만 쓴다고 해서 하이퍼텍스트가 만들어 지지 않는다.\
> ```<a href = "주소">``` 태그를 쓰면 주소와 연결되는 하이퍼텍스트가 만들어 진다.\
> ```<a href = "주소" target = "_blank">``` 같이 속성을 여러개 쓸수도 있다.
> 
```html
<a target = "_blank" href = "https://opentutorials.org/module/1892/10932">하이퍼텍스트</a>
```

#### 목록

```html
<li>HTML?</li>
<li>기본문법</li>
<li>포켓몬</li>
<li>디지몬</li>

<li>피카츄</li>
<li>라이츄</li>
<li>파이리</li>
<li>꼬북이</li>
```
> ## 어? 왜 두개가 붙어서 나오지? > ```<ul>``` 이나 ```<ol>``` 로 묶어주자!

```html
<ol> <!-- 순서가 있는 리스트 -->
    <li>HTML?</li>
    <li>기본문법</li>
    <li>포켓몬</li>
    <li>디지몬</li>
</ol>
<ul>
    <li>피카츄</li>
    <li>라이츄</li>
    <li>파이리</li>
    <li>꼬북이</li>
</ul>

<!-- 제대로 나온다 ㅎㅎ -->
```
