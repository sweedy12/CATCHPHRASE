
var link = "http://127.0.0.1:8000/items/";
var link_empty = "http://127.0.0.1:8000/";
var elementsInsideBody = [...document.body.getElementsByTagName("p")];




function pythonTest(){
	alert("here we go");
	start_request();
	//alert(elementsInsideBody);
	elementsInsideBody.forEach(element => {
                        test(element);
                })
	
}

function test(element){
	var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
		alert("were here");
      var obj = JSON.parse(this.responseText);
	  alert(obj.new_item);
	  element.innerHTML =  obj.new_item
    }
  };
	//var text = '{"text": element.innerHTML}'
	var data = JSON.stringify({"text": element.innerHTML});
	//js_res["text"] = element.innerHTML;
	//data.append("text",element.innerHTML);
	//alert(cur_link);
    xhttp.open("POST", link, true);
    xhttp.send(data);
}


function start_request(element){
	var xhttp = new XMLHttpRequest();
	var cur_link = link_empty;
	//alert(cur_link);
    xhttp.open("GET", cur_link, true);
    xhttp.send();
}


pythonTest();