import argparse
from flask import Flask, request, render_template, redirect, url_for
from waitress import serve
import os

from api import NercAPI


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    api = NercAPI()

    @app.route('/')
    def base():
        return redirect(url_for('nerc_home'))

    @app.route('/nerc_home', methods=['POST', 'GET'])
    def nerc_home():
        if request.method == 'GET':
            return render_template('index.html')
        else:
            input_text=request.form['input_text']
            result = api.get_nerc_result(input_text)
            return render_template('result.html', 
                               input_text=input_text,
                               nerc_output=result)

    return app

def run():
    parser = argparse.ArgumentParser(description='NERC Service')
    parser.add_argument('-p', '--port', help='port', type=int, default=9093)
    parser.add_argument('-host', '--host', help='host', default='localhost')
    args = parser.parse_args()
    app = create_app()
    # app.run(host=args.host, port=args.port)
    serve(app, host=args.host, port=args.port)

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    run()
