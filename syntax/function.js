function RETURNHELLO(HOWMUCH)
{
    i = 0;
    while(i < HOWMUCH)
    {
        console.log("HELLO\n");
        console.log(i+1);
        i++
    }

}

RETURNHELLO(10); ///10번 반복

console.log("\ns--------------------------")

function STRINGSUM (string1, string2)
{
    string3 = string1 + string2;

    return string3
}

console.log(STRINGSUM("YEEEEEE", "\nI HATE NODE.JS"));