#!/usr/bin/env bash

echo 'TensorBoard serving at http://localhost:8000'

tensorboard --logdir='assets/models/ssd' --port=8000
