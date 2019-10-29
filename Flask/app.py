from flask import Flask, render_template, request, redirect
from flask_restful import Resource, Api
from flask import flash
from function import *
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

		if message == '':
			print('Nothing input')
			return redirect(request.url)

		#image_path = data.filename
		#image_save = image_path.replace('las','csv')

		recommender = query_similar_books(message,10)
		titles = []
		ratings = []
		imgs = []

		for i in range (0,len(recommender)):
			titles.append(recommender.index[i][1])
			ratings.append(recommender.index[i][2])
			imgs.append(recommender.index[i][3])
			#print(name)
			#print(imgs)

		print(imgs)

		return render_template('results.html',
			title0 = titles[0], rating0 = ratings[0], img0 = imgs[0],
			title1 = titles[1], rating1 = ratings[1], img1 = imgs[1],
			title2 = titles[2], rating2 = ratings[2], img2 = imgs[2],
			title3 = titles[3], rating3 = ratings[3], img3 = imgs[3],
			title4 = titles[4], rating4 = ratings[4], img4 = imgs[4])

		selectedfile = ""

	return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)

