echo "Extracting frames from video file GoodGateTestLensCorrect.mp4..."

pip3 install -q opencv-python numpy tqdm

export CURR_DIR_CUSTOM=$(pwd)

python3 ./preprocessing/simple_gate_detector.py -v ./GoodGateTestLensCorrect.mp4 -o ./hough_lines_log.txt 

echo  "Executed OpenCV Gate Detector"

jupyter nbconvert --to notebook --execute ./preprocessing/preprocessing2.ipynb

echo "Executed preprocessing script"

git clone https://github.com/ultralytics/yolov5.git

echo "Cloning yolov5 git repo"

sed -i 's/nc: 80  # number of classes/nc: 1  # custom number of classes/g' ./yolov5/models/yolov5s.yaml

echo "Modified yolov5 classes"

rm ./gates/labels/*.cache

echo "Cleared cache files"

echo "Beginning training"
python3 ./yolov5/train.py --img 720 --batch 1 --epochs 5 --data ./qubo_gates.yaml --cfg ./models/yolov5s.yaml --weights '' 

echo "Training Complete. To view metrics, run tensorboard --logdir runs"

echo "Exporting path to best.pt wights file"

export WEIGHTS_DIR=${CURR_DIR_CUSTOM}"/runs/"$(cd $CURR_DIR_CUSTOM"/runs" && ls | tail -1)"/weights/best.pt"

echo "Running detection algorithm"

python3 ./yolov5/detect.py --source ./GoodGateTestLensCorrect.mp4 --weights $WEIGHTS_DIR
