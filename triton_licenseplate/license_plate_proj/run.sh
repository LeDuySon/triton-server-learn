test_image=$1

python3 client.py image ${test_image} --model yolov7lp_det_tensorrt --url localhost:8001 --width 416 --heigh 416