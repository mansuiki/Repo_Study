var fs = require("fs");

console.log("Est reiciendis quam iusto maiores alias eum maiores suscipit nulla.");

console.log("---------");

fs.readFile("syntax/Exam.txt", "utf8", function(err, result)
{
    if(err)
    {
        console.log(err);
    }
    console.log(result);
});

console.log("BYE________");



console.log("비동기 !!!!");

console.log("AABC");

var text = fs.readFileSync("syntax/Exam.txt", "utf8");

console.log(text+"비동기");

console.log("BYEBYE_____");