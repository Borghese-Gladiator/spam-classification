# Spam Classification using Text Features
Classification using SKLearn + NLTK for feature extraction

## Methodology
1. [Gathering Data](#gathering-data)
2. [Scraping Known Spam Words List](#scraping-known-spam-words-list)
3. [Preprocessing Text](#preprocessing-text)
4. Load CSV into numpy array (& change max limit for csv reader)
5. [Feature Extraction](#feature-extraction)
6. [Trained & Evaluated Classifiers](#trained-evaluated-classifiers)
8. Wrote best_classifier to /training_output/

## Gathering Data
Found dataset on Kaggle: [https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset](https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset)
- see archive.zip for data (spam_or_not_spam.csv is gitignored)
- Label of 1 means message is spam
- Label of 0 means message is NOT spam

## Preprocessing Text
**CHOSE TO OMIT STEP** - preprocessing takes WAY too long

Used NLTK for below preprocessing
- tokenization
- normalization (lemmatization)
- removing stop words

## Scraping Known Spam Words List
- Went to website & copied HTML from: [https://www.automational.com/spam-trigger-words-to-avoid/](https://www.automational.com/spam-trigger-words-to-avoid/)
- Wrote Vanilla JS script to iterate over elements & get value of each
- Logged whole object to console & saved array to Python list in utils.py.

JavaScript script included below
```html
<script>
  const result = []
  const ulElem = document.getElementById("BLAH");
  const items = ulElem.getElementsByTagName("li");
  for (let i = 0; i < items.length; ++i) {
    // do something with items[i], which is a <li> element
    const spanElem = items[i].getElementsByTagName("span")[1];
    result.push(spanElem.innerHTML)
  }

  console.log(JSON.stringify(result))
</script>
```

## Feature Extraction
- preprocessed text
  - tokenized, normalized (lemmatization), removed stop words
- Features Extracted (found with NLTK functions)
  - count of misspelled words
  - count of spam phrases based on scraped words
  - count of exclamation points

## Trained \& Evaluated Classifiers
- For every model (Decision Tree, Gradient Boost, and Random Forest Classifier)
  - ran 10-fold cross validation
  - calculated total accuracy, precision, and recall

## Initial Plan (Submitted)
- Gather dataset of spam and not spam
- Extract Features from spam text
  - Count of misspelled words
  - known spam phrases (eg: CLICK HERE)
  - Count of exclamation points
- Run Classification models (DecisionTree, GradientBoost, etc.)
  - Save best classifier & run classifier on sample test data
