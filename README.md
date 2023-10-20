# 1800-readme
---
## Project Description
### Facebook Programming Language Predictions
+++++++++++++++++\
\
**Overview:**\
Extracting most used langauge from facebook-research repositories, this will then be used to predict based on language used in the README what language is likely used. \
**Method**\
Classification using provided NLP data.\
**Takeaways**\
TBD.\
**Skills:**\
Python, Natural Language Processing (NLP), Pandas, EDA, Facebook, SciKit-Learn, Classification

## Project Goals
---
- Sufficient evidence to understand the frequently used languages and their associated context/words
- Classify the language accurately with respective models

## Initial Thoughts
--- 
There will likely be the same 4 languages that will be heavily used within the projects of the company, this will still provide insight as to what context is provided when the \
language is used.

## Planning
--- 
1. Acquire readmes from facebook-research repos
2. Clean said data
3. Explore and analyze data for better insights on relative context to language\
    a. Determine our baseline prediction\
        1. \
        2. \
        3. 

## Data Dictionary
--- 
| Feature        | Definition                                   |
| ---            | ---                                          |
| open  | the price of currency at the start of the reported time frame |
| close | **TARGET** the price of currency at the end of reported time frame |
| high | the highest price of the reported time frame |
| low   | the lowest price of the reported time frame |
| volume  | a cumulation of trade volume data from the largest volume exchanges |
| month   | month of the year |
| weekday   | a text indicator of what day of the week it is |


## Reproducability Requirements
---
1. Clone repo
RUN OPTIONS:\
    2a. Local machine must have python 3.7 and prophet downloaded\
    2b. Using Google Colab drop wrangle into files
3. Run notebook

## Conclusions 
- Volatility of Ethereum makes it very hard to predict seasonality / trend on a macro scale but with the recent
consistency there's sufficient data to be able to make a fairly close prediction
- The nature of price is that is a quasi-schotastic value that has lots of factors even outside that of which\
can be seen on a time scale. Such as news, improvements and maybe even downtime

## Recommendation
- Utilize other models outside of a time series -- ex. utilizing trade volume, open, high and low for a regression model
- Cross validation to improve model accuracy
