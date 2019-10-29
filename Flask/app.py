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
		#print(message)

		if message == '':
			print('Nothing input')
			return redirect(request.url)

		message1 =  message.split("\\")

		if (len(message1)>1):
			message = message1[0]
			not_message = message1[1]
		else:
			message = message1[0]
			not_message = ""

		#print(message1)
		#print(message)

		recommender = query_similar_books(message,20)
		not_recommender = query_similar_books(not_message,20)
		recommender = recommender[recommender.ensemble_similarity != 0]
		not_recommender = not_recommender[not_recommender.ensemble_similarity != 0]

		titles = []
		ratings = []
		imgs = []

		true_recommender = []
		for index in recommender.index:
			if index not in not_recommender.index:
				true_recommender.append(index)

		#print(true_recommender)

		for index in true_recommender:
			titles.append(index[1])
			ratings.append(index[2])
			imgs.append(index[3])

		#print(imgs)

		return render_template('results.html',
			title0 = titles[0], rating0 = ratings[0], img0 = imgs[0],
			title1 = titles[1], rating1 = ratings[1], img1 = imgs[1],
			title2 = titles[2], rating2 = ratings[2], img2 = imgs[2],
			title3 = titles[3], rating3 = ratings[3], img3 = imgs[3],
			title4 = titles[4], rating4 = ratings[4], img4 = imgs[4],
			title5 = titles[5], rating5 = ratings[5], img5 = imgs[5])

		selectedfile = ""

	return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)

