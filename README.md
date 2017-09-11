# Object Recognition Project
Object recognition for self driving. DS502-1702 capstone project track 1.
We use [SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow) as pre-trained model.

Team name: 乌波尔刷分团 \
Team member: Jin Sun, Xintong Xia

Installation:
```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Train: load pre-trained model, train, and save trained model
```
./scripts/run.sh train
```

Test: load trained model, batch predict on test set
```
./scripts/run.sh test
```

Serve:
```
./scripts/run.sh serve
```
