from flask import Flask, render_template, request, url_for
from GAN import GAN

app=Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
       tweets = request.form.get('ancien')
       gan = GAN(tweets)
       t = gan.entrainer()[0]
       return render_template('index.html', text=tweets, tweet=t)
    return render_template('index.html')
if __name__=="__main__":
    app.run(debug=True)