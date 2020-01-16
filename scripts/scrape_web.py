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
import csv
import pandas as pd


### SCRAPE AUTOTRADER ###

# stipulate search parameters
baseurl = 'https://www.autotrader.com/cars-for-sale/Used+Cars/'
location = 'Burlingame+CA-94010?'
listingtype = 'listingTypes=USED&' # listingTypes=USED%2CNEW&
radius = 'searchRadius=100&'
zipcode = 'zip=94010&'
includenew = 'marketExtension=include&isNewSearch=true&'
sorting = 'sortBy=relevance&numRecords=100&'
firstrecord = 'firstRecord='

for index in range(1000,2000,100):
    
    # build in wait time
    wait_time = npr.randint(10,25)
    time.sleep(wait_time)
    
    # print status update
    print('Scraping data for page with first listing index = %s ' % index)

    # compile full URL
    url = baseurl + location + listingtype + radius + zipcode + includenew + sorting + firstrecord + str(index)
    
    # get homepage session
    session = requests.Session()
    headers = {'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0'}
    
    # parse content
    homepage = session.get(url, headers=headers)
    tree = html.fromstring(homepage.content)
    
    # loop over all entries and collect this data
    listings = []
    for item in range(0,103):
        xpath = '/html/body/div[1]/div/div[2]/div[2]/div/div[2]/div[1]/div[2]/div[3]/div/div[2]/script[' + str(item) + ']/text()'
        listing_individual = tree.xpath(xpath)
        listings.append(listing_individual)
    
    # flatten list of lists
    listings = [item for sublist in listings for item in sublist]
    
    # # split on commas
    # newlist = []
    # for word in listings:
    #     word = word.split(",")
    #     newlist.extend(word)  # <----
    
    # create pandas dataframe
    listings_df = pd.DataFrame(listings)
    
    # save each batch of 100 listings
    listings_df.to_csv('listings_' + str(index) + '_to_' + str(index+100))














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
