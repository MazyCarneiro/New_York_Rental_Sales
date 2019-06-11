# New_York_Rental_Sales

The New York property sales dataset (Enigma, 2018) used in this assignment is derived from the Department of City Planning’s Primary Land Use Tax Lot Output 2017 dataset and the Department of Finance rolling sales dataset. The dataset was curated by Enigma by joining the two datasets via the address field. The property sales are only from the year 2017.

The purpose of this assignment is to test the hypotheses, where the classification goal was to predict whether the selling price of a property in 2017 was above(1) or(0) below the median value of the properties in New York in that same year (Manhattan, Brooklyn, Queens, Bronx and Staten Island.

The technologies used to carry out the analysis will be Hadoop which is an ecosystem and Spark which will help in executing tasks. The programming language used is Python. All the python code written to analyse the dataset was written in a .py file and include all the libraries needed for the analysis.

The CSV files have to be uploaded to Hadoop using the command <hadoop fs -copyFromLocal nameOfCSV.csv
The preprocessing is done in Preprocessing.py and the models are executed in the Big_Data_Applications_Coursework2.py .
