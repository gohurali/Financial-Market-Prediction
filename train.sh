#! /bin/bash
echo "-- Training Model --"
python3 train.py --save-model --onnx --output-dir models/
# END
