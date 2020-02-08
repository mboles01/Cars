#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:05:36 2020

@author: michaelboles
"""

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Depreciator/scripts') 

# import modules
from lxml import html
import requests
import time
import numpy.random as npr
import pandas as pd

### SCRAPE AUTOTRADER ###


# stipulate search parameters
radius = 'searchRadius=100&'
sorting = 'sortBy=distanceASC&'
number = 'numRecords=100&'
firstrecord = 'firstRecord='

# # SF
# location = 'San+Francisco+CA-94107?'
# zipcode = 'zip=94107&'

# # Chicago
# location = 'Chicago+IL-60616?'
# zipcode = 'zip=60616&'

# # New York
# location = 'New+York+NY-10101?'
# zipcode = 'zip=10101&'

# Los Angeles
location = 'Los+Angeles+CA-90012?'
zipcode = 'zip=90012&'

# # Houston
# location = 'Houston+TX-77002?'
# zipcode = 'zip=77002&'

# # search used only
# baseurl = 'https://www.autotrader.com/cars-for-sale/Certified+Cars/'
# isnew = 'isNewSearch=false&'
# listingtype = 'listingTypes=CERTIFIED%2CUSED&'
# includenew = 'marketExtension=include&listingTypes=USED%2CCERTIFIED&'

# search new only
baseurl = 'https://www.autotrader.com/cars-for-sale/New+Cars/' 
isnew = 'isNewSearch=true&'
listingtype = 'listingTypes=NEW&'
includenew = 'marketExtension=include&isNewSearch=true&'

    
for year in range(2020,2021,1):
    
    # print status update
    print('Scraping %s' % year)
    
    yearmade = str(year) + '/'
    startyear = 'startYear=' + str(year) + '&'
    endyear = 'endYear=' + str(year) + '&'

    for index in range(0,1100,100):
        
        # compile full URL
        # url = baseurl + yearmade + location + listingtype + radius + zipcode + startyear + includenew + endyear + isnew + sorting + number + firstrecord + str(index)        
        url = baseurl + yearmade + location + radius + zipcode + startyear + endyear + includenew + isnew + listingtype + sorting + number + firstrecord + str(index)        

        # print status update
        print('Page %s' % str(int(index/100+1)))
        # print(url)
    
        # build in wait time
        wait_time = npr.randint(0,2)
        time.sleep(wait_time)
    
        # get homepage session
        session = requests.Session()
        headers = {'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0'}
        
        # parse content
        homepage = session.get(url, headers=headers)
        tree = html.fromstring(homepage.content)
        
        # count number of results
        results = tree.xpath('//div[@class="results-text-container text-size-200"]//text()')
        print(results)
    
        # loop over all entries and collect this data
        listings = []
        for item in range(0,103):
            xpath = '/html/body/div[1]/div/div[2]/div[2]/div/div[2]/div[1]/div[2]/div[3]/div/div[2]/script[' + str(item) + ']/text()'
            listing_individual = tree.xpath(xpath)
            listings.append(listing_individual)
        
        # flatten list of lists
        listings = [item for sublist in listings for item in sublist]
        
        # create pandas dataframe
        listings_df = pd.DataFrame(listings)
        
        # save each batch of 100 listings
        listings_df.to_csv('../data/raw/Los_Angeles/New_2017-2020/' + str(year) + '_' + str(int(index/100+1)) + '.csv')









