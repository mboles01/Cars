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
from user_agent import generate_user_agent
import time
import numpy.random as npr
import pandas as pd

### SCRAPE AUTOTRADER ###

https://www.autotrader.com/cars-for-sale/New+Cars/2020/San+Francisco+CA-94107?listingTypes=NEW&searchRadius=100&zip=94107&startYear=2020&endYear=2020&marketExtension=include&isNewSearch=true&sortBy=distanceASC&numRecords=25&firstRecord=0

# stipulate search parameters
baseurl = 'https://www.autotrader.com/cars-for-sale/New+Cars/' # 'https://www.autotrader.com/cars-for-sale/Certified+Cars/'
location = 'San+Francisco+CA-94107?'
listingtype = 'listingTypes=NEW&' # 'listingTypes=CERTIFIED%2CUSED&' 
radius = 'searchRadius=100&' # 'searchRadius=300&'
zipcode = 'zip=94107&'
includenew = 'marketExtension=include&isNewSearch=true&' # 'marketExtension=include&'
# style = 'vehicleStyleCodes=SEDAN&'
isnew = 'isNewSearch=true&' # 'isNewSearch=false&'
sorting = 'sortBy=distanceASC&' # 'sortBy=mileageASC&' 'sortBy=yearDESC&'
number = 'numRecords=100&'
firstrecord = 'firstRecord='

for year in range(2017,2018,1):
    
    # print status update
    print('Scraping %s' % year)
    
    yearmade = str(year) + '/'
    startyear = 'startYear=' + str(year) + '&'
    endyear = 'endYear=' + str(year) + '&'

    for index in range(0,300,100):
        
        # compile full URL
        url = baseurl + yearmade + location + listingtype + radius + zipcode + startyear + includenew + endyear + isnew + sorting + number + firstrecord + str(index)        
        
        # print status update
        print('Page %s' % str(int(index/100+1)))
        print(url)
    
        # build in wait time
        wait_time = npr.randint(1,3)
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
        listings_df.to_csv('../data/raw/listings_' + str(year) + '_' + str(int(index/100+1)) + '.csv')














# pull listing information from webpage
tree.xpath('//div[@class="topRight"]//[div@class="first-price"]')

tree.xpath('//div[@class="first-price"]')

# titles = tree.xpath('//div[@class="display-flex justify-content-between"]//text()')
# prices = tree.xpath('//span[@class="first-price"]//text()')
# location_raw = tree.xpath('//div[@class="text-bold text-subdued"]//text()')
# location = location_raw[::2]
# distance = location_raw[1::2]
# other = tree.xpath('//ul[@class="list list-inline margin-bottom-0"]//text()')


listing_ids = list(OrderedDict.fromkeys(tree.xpath('//li[@class="component_property-card js-component_property-card js-quick-view "]//@data-propertyid')))
 
# get homepage session
url = 'https://www.realtor.com/realestateandhomes-search/' + zipcode + '/beds-1/baths-1/type-single-family-home'
session = requests.Session()
headers = {'User-Agent': generate_user_agent()}

#        proxies = get_proxies()
#        proxy_pool = cycle(proxies)
#        proxy = next(proxy_pool)
#        proxy = '110.232.80.234:4145'
#        try: 
#        homepage = session.get(url, timeout = 15, verify='Lam_certificate_Realtor_June2019_2.cer', headers = headers, proxies={"http": proxy, "https": proxy}) # Mac
#            print(homepage)
#        except:
#                Most free proxies will often get connection errors. You will have retry the entire request using another proxy to work. 
#                We will just skip retries as its beyond the scope of this tutorial and we are only downloading a single url 
#            print("Skipping. Connection error")
#        
homepage = session.get(url, timeout = 15, headers = headers) #, proxies={"http": proxy, "https": proxy}) # Mac
#        homepage = session.get(url, verify='Lam_certificate_Realtor_June2019.cer', timeout = 5, headers = headers) # PC
tree = html.fromstring(homepage.content)
#        soup = BeautifulSoup(homepage.content, "html.parser")

# update status
print('Scraping data for zipcode (%s/%s): ' % (counter,len(zipcodes)) + str(zipcode))
