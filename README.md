# rpn-for-faster_rcnn
Region Proposal Network Implementation for Faster RCNN
The RPN used to create outputs below was trained for only 40000 epochs instead of 80000.

![alt text](https://github.com/nathanjjohnson7/rpn-for-faster_rcnn/blob/main/results/bus_with_people.png?raw=true)

![alt text](https://github.com/nathanjjohnson7/rpn-for-faster_rcnn/blob/main/results/mountain_climber.png?raw=true)

![alt text](https://github.com/nathanjjohnson7/rpn-for-faster_rcnn/blob/main/results/people_on_street.png?raw=true)

![alt text](https://github.com/nathanjjohnson7/rpn-for-faster_rcnn/blob/main/results/people_running.png?raw=true)

![alt text](https://github.com/nathanjjohnson7/rpn-for-faster_rcnn/blob/main/results/horse_drinking_water.png?raw=true)

![alt text](https://github.com/nathanjjohnson7/rpn-for-faster_rcnn/blob/main/results/dog_running.png?raw=true)

The Faster RCNN is a combination of the Region Proposal Network and the Fast RCNN. 

To build a Faster RCNN model using 4-stage-training:

1.train the RPN (rpn-for-faster_rcnn/build_rpn.py)
2.get proposals from RPN (rpn-for-faster_rcnn/create_rpn_dataset.py)
3.train Fast RCNN with dataset (fast_rcnn/build_classifier.py)
4.train another RPN initialized with frozen base model weights from Fast RCNN
5.get proposals from new RPN
6.continue training Fast RCNN with new dataset (keep base model weights frozen)
