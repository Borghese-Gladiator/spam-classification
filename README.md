# Spam Classification using Text Features
Final project for CS 328

## Methodology
- [Gathering Data](#gathering-data)
- [Preprocessing Text](#preprocessing-text)
- [Scraping Known Spam Words List](#scraping-known-spam-words-list)
- Extracted Features using FeatureExtractor from features.py
- Trained & Evaluated classifiers
- Wrote best_classifier to /training_output/

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
- Wrote Vanilla JS script ([see below](#Script Snippet)) to iterate over elements & get value of each
- Logged whole object to console & saved array to Python script.

JS script included below
```
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

## Initial Plan (Submitted)
- Gather dataset of spam and not spam
- Extract Features from spam text
  - Count of misspelled words
  - known spam phrases (eg: CLICK HERE)
  - Count of exclamation points
- Run Classification models (DecisionTree, GradientBoost, etc.)
  - Save best classifier & run classifier on sample test data