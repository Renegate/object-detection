# Object Detection Project
Object detection for self driving. DS502-1702 capstone project track 1.
We use [SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow) as pre-trained model.

Team member: Jin Sun, Xintong Xia

For more details please look at our [Presentation Slides](https://docs.google.com/presentation/d/1zjTmVe6mQhGUx5bn145c0nYHdap9p5cSy_kiANv9StQ/edit?usp=sharing)

Installation:
```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Train: load pre-trained model, train (currently not implemented), and save trained model
```
./scripts/run.sh train
```
If you are running at first time, you have to running training commands before testing and serving,
because it downloads pretrained model.

Test: load trained model, batch predict on test set
```
./scripts/run.sh test
```

Serve: run model on a video
```
./scripts/run.sh serve
```
