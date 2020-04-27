import cherrypy
from initialize.init_models import VocabDict, WeightMatrix,GetEncoderDecoderTrainedModel
from predict.predictions import predict


class Application(object):
    @cherrypy.expose()
    def predict_sent(self,get_req):
        output = predict(get_req)
        print(output)
        return " ".join(output)

    @cherrypy.expose(alias="/")
    def get(self):
        return "APPLICATION STARTED"


if __name__ == '__main__':
    vocab_dict = VocabDict.get_vocab_instance()
    weight_matrix = WeightMatrix.get_instance()
    encoder, decoder = GetEncoderDecoderTrainedModel.get_instance()
    cherrypy.quickstart(Application())

