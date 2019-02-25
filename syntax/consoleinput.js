//node consoleinput.js 5 6 7 8 9 10 
//위의 명령을 터미널에서 실행시켜보자

process.argv.forEach(function (val, index, array) {
    console.log(index + ': ' + val);
  });


//결과 

/*
0: /usr/local/Cellar/node/11.10.0/bin/node
1: /Users/mansuiki/Documents/VisualStudioCode/NodeJS/syntax/consoleinput.js
2: 5
3: 6
4: 7
5: 8
6: 9
7: 10
*/

//즉, 터미널에서의 입력값을 처리 가능하다.