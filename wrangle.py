"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
# os/file access
import os
import json
from typing import Dict, List, Optional, Union, cast
# http request 
import requests
# arrays/df
import numpy as np
import pandas as pd
# viz
import seaborn as sns
import matplotlib.pyplot as plt
# random
import random
# nlp
from bs4 import BeautifulSoup
import unicodedata
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

from env import github_token, github_username


##########                                      ##########
##########              ACQUIRE                 ##########
##########                                      ##########

    # Define user agents for request headers
user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Mozilla/5.0 (Windows NT 10.0; rv:78.0) Gecko/20100101 Firefox/78.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0"
    ]

    # Randomly select a user agent
headers = {'User-Agent': random.choice(user_agents)}


# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.


def get_github_repos():
    
    base_url =  'https://github.com/orgs/facebookresearch/repositories'
    url_template = base_url + '?page={}'
    
    repo_links = []
    for page_num in range(1, 32):
        url = url_template.format(page_num)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        repo_links_on_page = [element['href'] for element in soup.find_all('a', itemprop='name codeRepository')]
        repo_links.extend(repo_links_on_page)
    return repo_links
    


# REPOS = get_github_repos()
 

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200: 
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_contents = requests.get(get_readme_download_url(contents)).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in get_github_repos()]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)
    
    
def process_all_repos():
    # Check if cached data exists
    if os.path.exists('all_facebook_repos.csv'):
        df = pd.read_csv('all_facebook_repos.csv')
        
        # Fill null values in language column with "Markdown"
        df = df.fillna({"language": "Markdown"})
        df = df.drop_duplicates(subset=['repo'],keep='first')
    else:
        # Get list of repo links
        repo_links = get_github_repos()

        # Process list of repo links and add to dataframe
        data = []
        for link in repo_links:
            out = a.process_repo(link)
            data.append(out)
        df = pd.DataFrame(data)

        # Cache processed data as CSV file
        df.to_csv('all_facebook_repos.csv', index=False)
        
        # Fill null values in language column with "Markdown"
        df = df.fillna({"language": "Markdown"})

    return df.astype({"repo": "string", "language": "string", "readme_contents": "string"})

##########                                      ##########
##########              PREPARE                 ##########
##########                                      ##########
def basic_clean(string):
    '''
    This function takes in the original text.
    The text is all lowercased, 
    the text is encoded in ascii and any characters that are not ascii are ignored.
    The text is then decoded in utf-8 and any characters that are not ascii are ignored
    Additionally, special characters are all removed.
    A clean article is then returned
    '''
    #lowercase
    string = string.lower()
    
    #normalize
    string = unicodedata.normalize('NFKD', string)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    string = string.replace('/',' ')
    string = string.replace('-',' ')
    #remove special characters and replaces it with blank
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    
    return string

def tokenize(string):
    '''
    This function takes in a string
    and returns the string as individual tokens put back into the string
    '''
    #create the tokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()

    #use the tokenizer
    string = tokenizer.tokenize(string, return_str = True)

    return string

def lemmatize(string):
    '''
    This function takes in a string
    and returns the lemmatized word joined back into the string
    '''
    #create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    #look at the article 
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    
    #join lemmatized words into article
    string = ' '.join(lemmas)

    return string

def remove_stopwords(string, extra_words = [], exclude_words = []):
    '''
    This function takes in text, extra words and exclude words
    and returns a list of text with stopword removed
    '''
    #create stopword list
    stopword_list = stopwords.words('english')
    
    #remove excluded words from list
    stopword_list = set(stopword_list) - set(exclude_words)
    
    #add the extra words to the list
    stopword_list = stopword_list.union(set(extra_words))
    
    #split the string into different words
    words = string.split()
    
    #create a list of words that are not in the list
    filtered_words = [word for word in words if word not in stopword_list]
    
    #join the words that are not stopwords (filtered words) back into the string
    string = ' '.join(filtered_words)
    
    return string

def transform_data(df, extra_stopwords= []):
    df = df.rename(columns={'readme_contents':'original'})
    # df['clean'] = cleaned and tokenized version with stopwords removed
    df['clean'] = df['original'].apply(basic_clean
                                      ).apply(tokenize
                                             ).apply(remove_stopwords, extra_words=extra_stopwords)
    # df['lematized'] = lemmatized version of clean data
    df['lematized'] = df['clean'].apply(lemmatize)
    
    # Split lemmatized strings into lists of words
    df['lematized'] = df['lematized'].apply(lambda x: x.split())
    
    # Drop words longer than 15 characters
    df['lematized'] = df['lematized'].apply(lambda x: [word for word in x if len(word) <= 15])
    
    # Join lists of words back into strings
    df['lematized'] = df['lematized'].apply(lambda x: ' '.join(x))
    
    # Drop rows where langauge is Jupyter Notebook
    df = df[df.language != 'Jupyter Notebook']
    
    valid_languages = ['Python', 'C++', 'Markdown']
    df.loc[~df['language'].isin(valid_languages), 'language'] = 'Other'
    
    # Drop unamed column
    df = df.drop(columns=['Unnamed: 0'])    
    return df



##########                                      ##########
##########              EXPLORE                 ##########
##########                                      ##########
def language_distributions():
    '''
    Showing all distributions of length of readme words.
    '''
    # Creating a list for our languages we want to be accessed
    languages = ['Python', 'C++', 'Other', 'Markdown','All']

    # Loop to check by language
    for language in languages:
        print(f'Checking distributions of {language}:')
        if language != 'All':
            sns.histplot(df.len_lematized[df.language == language])
            plt.xlabel(f'{language} ReadMe Len Distribution')
            plt.show()
        # Else indicates that we are looking for All
        else:
            sns.histplot(df.len_lematized)
            plt.show()
        print('+ + + + + + + + + + + + + + + +')