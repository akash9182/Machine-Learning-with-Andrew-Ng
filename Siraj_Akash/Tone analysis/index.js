var http = require('http');
var express = require('express');
var bodyPareser = require('body-parser');
var watson = require('watson-developer-cloud');

var app = express();
app.use(bodyPareser());

app.get('/', function(req, res){
	var html = '<form action="/" method="post">' +
				'Write something:' + 
				'<input type="text" name="userName" placeholder="...." />' +
				'<br>' +
				'button type="Submit</button>' +
				'</form>';
	res.send(html);
});

app.post('/', function(req, res){

    //Retrieve usr input and store as a variable
    var userName = req.body.userName;

    //Create a tone analyzer variable using credentials from your Bluemix account 
    //www.ibm.com/cloud-computing/bluemix/
    var tone_analyzer = watson.tone_analyzer({
        username: "c3954a35-4ac5-4c7f-b885-6babf7a4a5cb",
  		password: "HgNahMMiPgO3",
  		version: 'v3',
        version_date: '2017-04-14'
    });

    tone_analyzer.tone({ text: userName},
       	function(err, tone) {
        if (err)
            console.log(err);
        else
            //this is the JSON response (tone analysis) that you can view in terminal
            console.log(JSON.stringify(tone, null, 2));
    });
    res.send(html);
});

var server = http.createServer(app);
var port = process.env.PORT || 3000;
server.listen(port, function() {
    console.log('Express server running on *:' + port);
});