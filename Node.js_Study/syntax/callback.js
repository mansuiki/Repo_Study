var a = function()
{
    console.log("TEST");
}

function example(callback)
{
    callback();
}

example(a);

a();