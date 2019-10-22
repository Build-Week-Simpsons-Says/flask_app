"""This is the proof of concept for attempting to connect the DS model to the
website"""

# Imports
from flask import Flask, request

def create_app():
    """create and configures an instance of a flask app"""
    app = Flask(__name__)

    @app.route('/')
    def root():
        return "App home page"
    return app

    @app.route('/input', methods=['POST'])
    @app.route('/input/<text>', methods=['GET'])
    def text():
        test_text = text
        return "You tried {} and the model gave it back".format(test_text)
