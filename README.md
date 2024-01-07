
# Organic Product Purchace Prediction Project



## About The Project

This project aims to predict whether a user will purchase an organic product based on various input features. Leveraging machine learning algorithms, the model analyzes user data and provides a binary prediction: 'Will Buy' or 'Will Not Buy'. The goal is to assist businesses in understanding customer behavior and optimizing marketing strategies.

## About the Data

1. Gender:
    - Data Type: Object
    - Insight: Records the gender of customers, offering insights into gender-specific purchasing patterns and preferences.
2. Geographic Region:
    - Data Type: Object
    - Insight: Captures the regional location of customers, enabling analysis of regional variations in buying behavior and product preferences.
3. Loyalty Status:
    - Data Type: Object
    - Insight: Indicates the loyalty tier of customers within the store's program, potentially influencing purchase frequency and   brand loyalty.
4. Neighborhood Cluster-55 Level:
    - Data Type: Float64
    - Insight: Assigns customers to specific neighborhood clusters, aiding in localized marketing strategies and understanding community preferences.
5. Neighborhood Cluster-7 Level:
    - Data Type: Object
    - Insight: Categorizes customers into broader neighborhood clusters, providing a macro view of community characteristics and potential market segments.
6. Television Region:
    - Data Type: Object
    - Insight: Identifies the television viewing region for customers, offering insights into regional media consumption habits and advertising opportunities.
7. Affluence Grade:
    - Data Type: Int32
    - Insight: Assigns a grade reflecting the affluence level of customers, helping to tailor marketing efforts and product offerings to specific economic segments.
8. Age:
    - Data Type: Int32
    - Insight: Records the age demographics of customers, facilitating age-specific marketing strategies and understanding generational buying behaviors.
9. Loyalty Card Tenure:
    - Data Type: Int32
    - Insight: Measures the longevity of customer participation in the loyalty program, indicating customer retention and long-term engagement levels with the store.


### Target variable: 
10. ORGANICS : Target variable indicating whether a customer is likely to purchase an organic product (1 for 'Will Buy', 0 for 'Will Not Buy').
  

Dataset Source Link : [https://www.kaggle.com/datasets/papercool/organics-purchase-indicator]




## Table of Contents

- [Organic Product Purchace Prediction Project]
(#Organic Product Purchace Prediction Project)
  - [About The Project](#about-the-project)
  - [About the Data](#about-the-data)
    - [Target variable:](#target-variable)
  - [Table of Contents](#table-of-contents)
  - [Installation and Dependencies](#installation-and-dependencies)
  - [Working Directory](#working-directory)
  - [Working with the code](#working-with-the-code)
  - [Contributing](#contributing)
  - [Contact](#contact)


## Installation and Dependencies

These are some required packages for our program which are mentioned in the Requirements.txt file

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- catboost
- xgboost
- Flask
- dill



## Working Directory

```
📦Housing_predict_recommend
 ┣ 📂artifacts
 ┃ ┗ 📜data.csv
 ┃ ┣ 📜model.pkl
 ┃ ┣ 📜preprocessor.pkl
 ┃ ┣ 📜test.csv
 ┃ ┣ 📜train.csv
 ┣ 📂catboost_info
 ┣ 📂logs
 ┣ 📂NOTEBOOK
 ┃ ┗ 📜EDA_OrganicStore.ipynb
 ┃ ┗ 📜Model_Trainer.ipynb
 ┣ 📂src
 ┃ ┣ 📂components
 ┃ ┃ ┣ 📜data_ingestion.py
 ┃ ┃ ┣ 📜data_transformation.py
 ┃ ┃ ┣ 📜model_trainer.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂pipeline
 ┃ ┃ ┣ 📜predict_pipeline.py
 ┃ ┃ ┣ 📜train_pipeline.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📜exception.py
 ┃ ┣ 📜logger.py
 ┃ ┣ 📜utils.py
 ┃ ┗ 📜__init__.py
 ┣ 📂static
 ┃ ┣ 📂css
 ┃ ┃ ┗ 📜homenew.css
 ┃ ┃ ┗ 📜resultnew.css
 ┃ ┗ 📂img
 ┃ ┃ ┣ 📜sucess.webp
 ┃ ┃ ┣ 📜failure.webp
 ┣ 📂templates
 ┃ ┣ 📜index.html
 ┃ ┣ 📜result.html
 ┣ 📜.gitignore
 ┣ 📜.gitattributes
 ┣ 📜app.py
 ┣ 📜main.py
 ┣ 📜README.md
 ┣ 📜requirements.txt
 ┗ 📜setup.py
 ```


## Working with the code


I have commented most of the neccesary information in the respective files.

To run this project locally, please follow these steps:-

1. Clone the repository:

   ```shell
   git clone https://github.com/sixface99ashwin/Organic-SuperStore-ML-Prediction.git
   ```


2. **Create a Virtual Environment** (Optional but recommended)
  It's a good practice to create a virtual environment to manage project dependencies. Run the following command:
     ```shell
     conda create -p <Environment_Name> python==<python version> -y
     ```

3. **Activate the Virtual Environment** (Optional)
   Activate the virtual environment based on your operating system:
      ```shell
      conda activate <Environment_Name>/
      ```

4. **Install Dependencies**
   - Navigate to the project directory:
     ```
     cd [project_directory]
     ```
   - Run the following command to install project dependencies:
     ```
     pip install -r requirements.txt
     ```

   Ensure you have Python installed on your system (Python 3.8 or higher is recommended).<br />
   Once the dependencies are installed, you're ready to use the project.



5. Run the Flask app: Execute the following code in your terminal.
   ```shell  
   python app.py 
   ```
   

6. Access the app: Open your web browser and navigate to http://127.0.0.1:5000/ to use the House Price Prediction and Property Recommendation app.




## Contributing
I welcome contributions to improve the functionality and performance of the app. If you'd like to contribute, please follow these guidelines:

1. Fork the repository and create a new branch for your feature or bug fix.

2. Make your changes and ensure that the code is well-documented.

3. Test your changes thoroughly to maintain app reliability.

4. Create a pull request, detailing the purpose and changes made in your contribution.

## Contact

Arumugam K - [aruashwin04@gmail](mailto:aruashwin04@gmail.com)



