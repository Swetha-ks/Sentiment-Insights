import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from io import BytesIO
from PIL import Image
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# ---------------- CONFIG ----------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment_analysis.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your_secret_key'

db = SQLAlchemy(app)

# Create uploads folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------- NLTK ----------------
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# ---------------- LOAD MODEL ----------------
# ⚠️ Make sure this file exists in your project folder
facial_emotion_model = load_model('facialdetection.h5')

emotion_classes = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"
}

# ---------------- DATABASE MODEL ----------------
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    content_type = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(10), nullable=False)
    image_path = db.Column(db.String(150), nullable=True)

with app.app_context():
    db.create_all()

# ---------------- HELPER FUNCTIONS ----------------
def preprocess_image(image):
    img = Image.open(image).convert('L')
    img = img.resize((48, 48))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict_emotion(image):
    img = preprocess_image(image)
    predictions = facial_emotion_model.predict(img)
    return emotion_classes[np.argmax(predictions)]

def predict_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    return 'Neutral'

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    posts = Post.query.all()
    return render_template('index.html', posts=posts)

@app.route('/upload_content', methods=['GET', 'POST'])
def upload_content():
    if request.method == 'POST':
        content_type = request.form['content_type']

        # -------- TEXT --------
        if content_type == 'text':
            text = request.form['text_content']
            sentiment = predict_sentiment(text)

            new_post = Post(
                user_id=1,
                content_type='text',
                content=text,
                sentiment=sentiment
            )
            db.session.add(new_post)
            db.session.commit()

        # -------- IMAGE --------
        elif content_type == 'image':
            if 'image_content' not in request.files:
                return redirect(request.url)

            file = request.files['image_content']
            if file.filename == '':
                return redirect(request.url)

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            image_data = BytesIO(open(file_path, 'rb').read())
            emotion = predict_emotion(image_data)

            new_post = Post(
                user_id=1,
                content_type='image',
                content=filename,
                sentiment=emotion,
                image_path=filename
            )
            db.session.add(new_post)
            db.session.commit()

        # ✅ Redirect to home after upload
        return redirect(url_for('index'))

    return render_template('upload_content.html')

@app.route('/post/<int:post_id>')
def view_post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('view_post.html', post=post)

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)