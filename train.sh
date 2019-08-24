#! /bin/bash
echo "-- Training Model --"
python3 train.py --save-model --onnx --ouput-dir models/
# END
