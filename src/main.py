import argparse
from flask import Flask, request, render_template, redirect, url_for
from waitress import serve
import os
from flask_ngrok import run_with_ngrok
import json
from api import NercAPI


def create_app(host, port):
    app = Flask(__name__, instance_relative_config=True)
    run_with_ngrok(app)
    api = NercAPI()

    @app.route('/')
    def base():
        return redirect(url_for('nerc_home'))

    @app.route('/nerc_home', methods=['POST', 'GET'])
    def nerc_home():
        if request.method == 'GET':
            return render_template('index.html', host=request.url)
        else:
            input_text=request.form['input_text']
            result = api.get_nerc_result(input_text)
            return render_template('result.html', 
                                   input_text=input_text,
                                   output00=json.dumps(result[0][0]),
                                   output01=json.dumps(result[0][1]),
                                   output10=json.dumps(result[1][0]),
                                   output11=json.dumps(result[1][1]),
                                   output2=json.dumps(result[2]),
                                   output30=json.dumps(result[3][0]),
                                   output31=json.dumps(result[3][1]),
                                   output4=json.dumps(result[4]),
                                   host=request.url)

    return app

def run():
    parser = argparse.ArgumentParser(description='NERC Service')
    parser.add_argument('-p', '--port', help='port', type=int, default=9093)
    parser.add_argument('-host', '--host', help='host', default='localhost')
    args = parser.parse_args()
    app = create_app(args.host, args.port)
    app.run()
    # serve(app, host=args.host, port=args.port)

if __name__ == '__main__':
    run()
