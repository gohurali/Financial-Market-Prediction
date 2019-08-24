import onnx
from onnx_tf.backend import prepare
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', 
                    type=str, 
                    #dest='input_dir'
                    default='outputs',
                    help='Gets model in a given location')
parser.add_argument('--output-path', 
                    type=str,
                    #dest='output_path' 
                    default='outputs',
                    help='Saves model in a given location')
args = parser.parse_args()


onnx_model = onnx.load(args.input_path)  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph(args.output_path)  # export the model