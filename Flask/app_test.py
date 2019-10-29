from flask import Flask, render_template, request, redirect
from flask_restful import Resource, Api
from flask import flash
from werkzeug.datastructures import FileStorage


app = Flask(__name__)
api = Api(app)



@app.route("/", methods=['POST', 'GET'])
def index():
	if request.method == 'POST':
		#print(request.form)
		if 'text' not in request.form:
			#flash('No file part')
			return redirect(request.url)

		message = request.form['text']
		print(message)

		message1 =  message.split("\\")

		if (len(message1)>1):
			message = message1[0]
			not_message = message1[1]
		else:
			message = message1[0]
			not_message = ""

		print(message1)
		print(message)

		if message == '':
			print('Nothing input')
			return redirect(request.url)

	selectedfile = ""

	return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)

