var Folder = "./data";
var fs = require("fs");

fs.readdir(Folder, function(err, filelist)
{
    if(err) 
    {
        console.log("YEEEEEE ERROR!!! \n");
        console.log(err);
    }
    console.log(filelist);
})