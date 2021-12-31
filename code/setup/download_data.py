import os
import sys

if __name__ == '__main__':
    # using os.mkdir caused the directory to made in the wrong disk & ran out of space really quicky... 
    # Not sure why, but this doesn't seem to be a problem when I use the OS to do it
    os.system('mkdir ../data') 
    # os.mkdir('data/')

    # refer to https://github.com/github/CodeSearchNet for data souring 
    os.system('curl https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip -o ../data/python.zip')
    os.system('curl https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip -o ../data/go.zip')
    os.system('curl https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip -o ../data/java.zip')
    os.system('curl https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip -o ../data/javascript.zip')
    print('done downloading data')

    os.system('unzip ../data/python.zip -d ../data/')
    os.system('unzip ../data/go.zip -d ../data/')
    os.system('unzip ../data/java.zip -d ../data/')
    os.system('unzip ../data/javascript.zip -d ../data/')
