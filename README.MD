# Classification of Bean Leaves 🌱🫘

This project develops a web interface using Django, which classifies images of bean leaves to identify issues such as spots, rust, or determine if the leaf is healthy. The system integrates Mistral, a Large Language Model (LLM), which generates explanatory responses based on the classification results, providing a detailed diagnosis of the plant's condition. The project aims to facilitate early disease detection, offering an interactive and accessible solution for agricultural monitoring.



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