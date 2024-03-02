from flask import Flask, render_template
import config

app = Flask(__name__)

@app.route('/')
def access():
    return render_template('access.html')


@app.route('/registration')
def registration():
    return render_template('registration.html')

##################################################################################################################

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)