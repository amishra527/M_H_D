#!/bin/bash

# Download the dataset from Kaggle
kaggle datasets download sudipchakrabarty/kiit-mita
kaggle datasets download rawsi18/military-assets-dataset-12-classes-yolo8-format

# Unzip the downloaded file
unzip kiit-mita.zip
unzip archive.zip