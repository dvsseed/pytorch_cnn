#!/bin/bash

echo "To execute our model..."
python cnn_l3_battery_ourmodel.py
sleep 60s

echo "To execute ResNet-18..."
python cnn_l3_battery_resnet18.py
sleep 60s

echo "To execute ResNet-34..."
python cnn_l3_battery_resnet34.py
sleep 6s

echo "All done."
