# Deals On Wheels: 
## Let the market show you how to buy a better car
Your car is almost certainly the largest depreciating asset you’ll ever buy. Unfortunately, the resources available to consumers to help inform their car buying decision have serious limitations: they are often human-curated, feature a small subset of what’s available, and do not support their claims with data. As a Data Science Fellow at Insight Data Science, I used Autotrader listings to develop an app that maps out car depreciation costs across hundreds of models and makes a recommendation to the user that will minimize their costs and maximize their satisfaction. 

Website: www.dealsonwheels.live

Slides: [Google Slides](https://docs.google.com/presentation/d/1KNRXbkX_s0U0u-Yd6xVhzvTzUBxkICbdNbBvkWmaS_M/edit#slide=id.g7d6a2fc1e6_0_0)


## Files

### `./scripts/`

### Web scraping
* `scrape_web.py`: uses *Requests* to connect to Autotrader, *html* to get web content, and *Pandas* to clean and store scraped content as a dataframe

* `clean_data.py`: these files load .csv files, concatenate dataframes, pull out data of interest, rename/reorder columns, and remove spurious listings. 

### Exploratory data analysis and fitting
* `analyze_data.py`: these files pull .csv file with listing information, create histogram and scatter plots, fit price data to car age and mileage across all make/model combinations, and plot 2D and 3D depreciation curves and box plots.

### `./flaskapp/flaskexample/`

### Web app development

* `views.py`: loads listing data and defines @app.route functions for /index, /models, /output, /random, and /about web pages.

* `.flaskapp/flaskexample/templates/`: contains html template files for the web pages listed above.



### Libraries
* [Requests](https://2.python-requests.org/en/master/)
* [Html](https://pypi.org/project/html/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Statsmodels](https://www.statsmodels.org/stable/index.html)
* [Scikit-learn](https://scikit-learn.org/stable/)

### Acknowledgement
Written by **Michael Boles** in February 2020 with help from the *Insight* and *StackOverflow* communities.
