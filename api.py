from flask import Flask, Blueprint
from flask import request, url_for
import urllib.request
from flask_restplus import Api, Resource, fields
import operator
import werkzeug
from werkzeug.datastructures import FileStorage
from flask_cors import CORS
from fast_bert.prediction import BertClassificationPredictor
from waitress import serve

MODEL_PATH = "./models/output/model_out"
LABEL_PATH = "./labels"
predictor = BertClassificationPredictor(
				model_path=MODEL_PATH,
				label_path=LABEL_PATH, # location for labels.csv file
				multi_label=True,
				model_type='xlnet',
				do_lower_case=False,
				device=None) # set custom torch.device, defaults to cuda if available


api_v1 = Blueprint('api', __name__, url_prefix='/api')

api = Api(api_v1, version='1.0', title='AI API',
    description='AI API',
)

ns = api.namespace('v1/', description='AI operations')

tc_parser = api.parser()
tc_parser.add_argument('text', type=str, required=True, help='text input', location='form')

@ns.route('/text_classification')
class TextClassification(Resource):
    @api.doc(parser=tc_parser)
    def post(self):
        '''text classification'''
        args = tc_parser.parse_args()
        input = args['text']
        single_prediction = predictor.predict(input)

        predict_array = []

        for val in single_prediction:
            predict_array.append(val[1])

        (m, i) = max((v, i) for i, v in enumerate(predict_array))
        dict = {}
        dict['probability'] = m
        dict['predict_label'] = single_prediction[i][0]
        dict['prediction'] = single_prediction

        return dict, 201


if __name__ == '__main__':
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(api_v1)
    serve(app, host='0.0.0.0', port=8008)
    # app.run(debug=True, host='0.0.0.0', port=5005)

