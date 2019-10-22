"""This is the proof of concept for attempting to connect the DS model to the
website"""

# Imports
from flask import Flask, Response, request, jsonify
from .predict import *

error_msg = f'''
            Something broke!
            Use the / or /years endpoints to get a list
            of possible values.
            '''


def create_app():
    """create and configures an instance of a flask app"""
    app = Flask(__name__)

    @app.route('/')
    def root():
        return "App home page"

    # @app.route('/input', methods=['POST'])
    @app.route('/input', methods=['GET'])
    def retrieval():
        try:
            if request.method == 'GET':
                text = request.args.get('input_text')  # If no key then null
                output = get_quote(text)
                return output  # This is now the input variable into the model
        except Exception as e:
            # Unfortunately I'm not going to wrap this in indv. strings
            r = Response(response=error_msg+str(e),
                         status=404,
                         mimetype="application/xml")
            r.headers["Content-Type"] = "text/xml; charset=utf-8"
            return r

    return app
