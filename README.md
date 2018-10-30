# API_Project-Quandl

This exercise will require to pull some data from the Quandl API. 

## Data

Qaundl is currently the most widely used aggregator of financial market data. In this project, data will be pull from the Quandl API.

At first, you will need to register a free account on the http://www.quandl.com website.

After you register, you will be provided with a unique API key, that you should store:
Store the API key as a string - according to PEP8, constants are always named in all upper case
API_KEY = ''

Qaundl has a large number of data sources, but, unfortunately, most of them require a Premium subscription. Still, there are also a good number of free datasets.

## Focus

For this project, we will focus on equities data from the Frankfurt Stock Exhange (FSE), which is available for free. We'll try and analyze the stock prices of a company called Carl Zeiss Meditec, which manufactures tools for eye examinations, as well as medical lasers for laser eye surgery: https://www.zeiss.com/meditec/int/home.html. The company is listed under the stock ticker AFX_X.

## Tasks

Project tries to find answers for the following questions:

1. Collect data from the Franfurt Stock Exchange, for the ticker AFX_X, for the whole year 2017 (keep in mind that the date format is YYYY-MM-DD).

2. Convert the returned JSON object into a Python dictionary.

3. Calculate what the highest and lowest opening prices were for the stock in this period.

4. What was the largest change in any one day (based on High and Low price)?

5. What was the largest change between any two days (based on Closing Price)?

6. What was the average daily trading volume during this year?

7. (Optional) What was the median trading volume during this year. (Note: you may need to implement your own function for calculating the median.)

Keep in mind that the JSON responses you will be getting from the API map almost one-to-one to Python's dictionaries. Unfortunately, they can be very nested, so make sure you read up on indexing dictionaries in the documentation provided below.

## Resources

Detailed Quandl API instructions here: https://docs.quandl.com/docs/time-series

Requests package, which can be easily downloaded using pip or conda: http://docs.python-requests.org/en/master/

Python Standard Library (the collections module): https://pymotw.com/3/collections/)

Data structures:  https://docs.python.org/3/tutorial/datastructures.html



