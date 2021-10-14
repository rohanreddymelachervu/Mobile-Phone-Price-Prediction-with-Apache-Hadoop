#!/usr/bin/env python3
"""mapper.py"""
import sys
count = 0
for line in sys.stdin:
    line = line.strip()
    print('%s\t' % count,line)
    count = count+1
  
