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
    b. Determine our stop words\
    c. View most commomn ngrams\
    d. Use stats testing\ 
4. Model using TFIDF
5. Document conclusions, recommendations, and next steps 

## Data Dictionary
--- 
| Feature        | Definition                                   |
| ---            | ---                                          |
| Unigrams  | Most used words by programing languages |
| Bigrams | Most common words grouped by 2 per programing languages |
| Trigram | Most common words grouped by 3 per programing languages  |



## Reproducability Requirements
---
1. Clone repo
2. Run notebook

## Conclusions 
- Because Facebook's github repos are primaraly python, our model is beneficial to recuters for the Facebook reasearch team. 
- Beat baseline and can accuracy predict python 
- Research teams success depends on their ability to collaborate. 
- Potential candidate’s GitHub’s can be vetted on their knowledge and experience with top programing languages. 
- Add more repositories form different departments to accurately predict programing languages from any department 

## Recommendation
- Show facebook research team recruiters the most popular programing languages. 
- Recruiters can use this to scan an applicants GitHub to see if they can be placed in a facebook research team
- Compare to other departments and see how this model performs 
