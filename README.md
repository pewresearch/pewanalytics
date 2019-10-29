# pewanalytics

`pewanalytics` is a Python package that provides text processing and statistics utilities for computational social science researchers.


## Installation 

Install with pip: 

    pip install https://github.com/pewresearch/pewanalytics#egg=pewanalytics

Install from source: 

    python setup.py install


## Instructions

    from pewanalytics import pewtext.TextCleaner

## Methodological Note

Pew Research Center


## Acknowledgements

Pew Research Center is a subsidiary of The Pew Charitable Trusts, its primary funder. This report is a collaborative effort based on the input and analysis of [a number of individuals and experts at Pew Research Center](link here).


## Use Policy 

In addition to the [license](https://github.com/pewresearch/pewanalytics/blob/master/LICENSE), Users must abide by the following conditions:

- User may not use the Center's logo
- User may not use the Center's name in any advertising, marketing or promotional materials.
- User may not use the licensed materials in any manner that implies, suggests, or could otherwise be perceived as attributing a particular policy or lobbying objective or opinion to the Center, or as a Center endorsement of a cause, candidate, issue, party, product, business, organization, religion or viewpoint.


# About Pew Research Center

Pew Research Center is a nonpartisan fact tank that informs the public about the issues, attitudes and trends shaping the world. It does not take policy positions. The Center conducts public opinion polling, demographic research, content analysis and other data-driven social science research. It studies U.S. politics and policy; journalism and media; internet, science and technology; religion and public life; Hispanic trends; global attitudes and trends; and U.S. social and demographic trends. All of the Center's reports are available at [www.pewresearch.org](http://www.pewresearch.org). Pew Research Center is a subsidiary of The Pew Charitable Trusts, its primary funder.


## Contact

For all inquiries, please email info@pewresearch.org. Please be sure to specify your deadline, and we will get back to you as soon as possible. This email account is monitored regularly by Pew Research Center Communications staff.

# Test data

Test data was downloaded from NLTK on 9/12/19:
```
import nltk
nltk.download("movie_reviews")
reviews = [{"fileid": fileid, "text": nltk.corpus.movie_reviews.raw(fileid)} for fileid in nltk.corpus.movie_reviews.fileids()]
reviews = pd.DataFrame(reviews)
reviews.to_csv("test_data.csv", encoding="utf8")
```