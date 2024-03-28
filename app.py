from flask import Flask, render_template, request, send_file
import os
import config
import sqlite3
from io import BytesIO
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy.exc import IntegrityError
from ML_test import ML_Match


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(50), nullable=False)
    surname = db.Column(db.String(50), nullable=False)
    iris_picture = db.Column(db.LargeBinary)


@app.route('/')
def access():
    return render_template('access.html')


@app.route('/registration')
def registration():
    return render_template('registration.html')

##################################################################################################################

@app.route('/signup', methods=['POST'])
def signup():
    try:
        email = request.form.get('email')
        name = request.form.get('name')
        surname = request.form.get('surname')
        iris_picture = request.files.get('iris_picture')

        if iris_picture:
            iris_picture_data = iris_picture.read()
        else:
            return 'Iris picture not provided.'

        new_user = User(email=email, name=name, surname=surname, iris_picture=iris_picture_data)
        db.session.add(new_user)
        db.session.commit()

        return 'Registration successful!'

    except IntegrityError as e:
        db.session.rollback()
        print(f"Error during registration - IntegrityError: {str(e)}")
        return 'Email address already in use. Please use a different email.'

    except Exception as e:
        db.session.rollback()
        print(f"Error during registration: {str(e)}")
        return f'Error during registration: {str(e)}'


##################################################################################################################
    

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    iris_picture_1 = request.files.get('iris_picture')
    user = User.query.filter_by(email=email).first()
    if user:
        # Connect to the database
        conn = sqlite3.connect('instance/site.db')
        cursor = conn.cursor()

        # Execute the query to retrieve iris_picture associated with the id
        cursor.execute('SELECT iris_picture FROM User WHERE email = ?', (email,))
        result = cursor.fetchone()

        # Close the database connection
        conn.close()

        iris_picture_blob = result[0]

        # Convert the blob into an image object
        iris_picture_2 = Image.open(BytesIO(iris_picture_blob))

        # Save the image in BMP format
        image = Image.open(iris_picture_1)
        file_name_1 = "data/iris_picture.bmp"
        image.save(file_name_1)

        # Save the image in BMP format
        file_name_2 = f'data/iris_image_{email}.bmp'
        iris_picture_2.save(file_name_2, 'BMP')

        bool = ML_Match(file_name_1, file_name_2)

        os.remove(file_name_1)
        os.remove(file_name_2)

        if bool:
            return 'Login successful!'
        else:
            return 'Iris matching failed. Try again.'
    else:
        return 'Email not found. Please sign up.'
    

##################################################################################################################
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)