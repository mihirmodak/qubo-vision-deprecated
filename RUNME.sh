git clone https://github.com/ultralytics/yolov5.git

cp ./yolov5/models/yolov5s.yaml ./yolov5/models/yolov5s_v2.yaml
sed -i 's/nc: 80  # number of classes/nc: 1  # custom number of classes/g' ./yolov5/models/yolov5s_v2.yaml
rm ./gates/labels/*.cache

python3 ./yolov5/train.py --img 720 --batch 3 --epochs 200 --data ./qubo_gates.yaml --cfg ./models/yolov5s.yaml --weights''

tensorboard --logdir runs

echo  "To run the model on the video, run: 'python3 ./yolov5/detect.py --source ./GoodGateTestLensCorrect.mp4 --weights PATH_TO_BEST_WEIGHTS', replacing PATH_TO_BEST_WEIGHTS with the actual path shown at the end of training"
