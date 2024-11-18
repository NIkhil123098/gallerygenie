from flask import Flask
@app.route("/getdata", methods=["POST"])
def getdata():
    return "Hello World"