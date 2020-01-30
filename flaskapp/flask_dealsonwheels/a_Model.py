#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:32:36 2020

@author: michaelboles
"""


def ModelIt(fromUser  = 'Default', births = []):
  in_month = len(births)
  print('The number born is %i' % in_month)
  result = in_month
  if fromUser != 'Default':
    return result
  else:
    return 'check your input'