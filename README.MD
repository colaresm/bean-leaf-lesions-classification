# Classification of Bean Leaves 🌱🫘

This project develops a web interface using Django, which classifies images of bean leaves to identify issues such as spots, rust, or determine if the leaf is healthy. The system integrates Mistral-7B-OpenOrca, a Large Language Model (LLM), which generates explanatory responses based on the classification results, providing a detailed diagnosis of the plant's condition. The project aims to facilitate early disease detection, offering an interactive and accessible solution for agricultural monitoring.

# About Classification
A MobileNet was trained using the (Bean Leaf Lesions Classification dataset)[https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification], which includes the categories "healthy", "angular leaf spot", and "bean rust". The dataset contains 1034 images in the training set and 133 images in the calibration set, totaling 155 MB. This dataset can be used for training and evaluating machine learning models aimed at recognizing lesions on legume leaves.


# Get Started 🚀
- To run the application, it is recommended to use a Python environment as some libraries will need to be installed.Initially, run the following command:

```
pip install requirements.txt
```
- To start the application, simply execute the following command:
```
python manage.py run server
```
The following theme will be displayed, then simply upload an image you wish to evaluate. The system includes a validation to ensure that only images of bean leaves are evaluated, preventing incorrect results from being generated.