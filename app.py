from flask import Flask, render_template, request
import config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy.exc import IntegrityError

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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)