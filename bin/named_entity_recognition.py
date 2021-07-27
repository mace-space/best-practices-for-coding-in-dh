#!/usr/bin/env python
# coding: utf-8

# # Named Entity Recognition of the John Henslow Collection
#
# ---
# ---
#
# ## What is Named Entity Recognition (NER)?
# The purpose of **named entity recognition** is to extract information from unstructured text, especially where it's impractical to have humans read and markup a large number of documents.
#
# A named entity can be any type of real-world object or meaningful concept that is assigned a name or a [proper name](https://en.wikipedia.org/wiki/Proper_noun#Proper_names). Typically, named entities can include:
#
# - people
# - organisations
# - countries
# - languages
# - locations
# - works of art
# - dates
# - times
# - numbers
# - quantities
#
# <p style="text-align: center; font-style: italic;">Examples of possible entities: a person, an organisation and a language (modern or ancient):
#     </p>
# <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Punjabi_woman_smile.jpg" alt="She was waiting for her turn to participate in 'Gidha' which is the folk dance from the state of Punjab, India, during the Youth Festival in Amritsar. Smile! You Are On! Raminder pal Singh: CC BY-SA 2.0" title="She was waiting for her turn to participate in 'Gidha' which is the folk dance from the state of Punjab, India, during the Youth Festival in Amritsar. Smile! You Are On! Raminder pal Singh: CC BY-SA 2.0" width="100" style="float: left; padding: 10px;">
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/World_Health_Organization_Logo.svg/320px-World_Health_Organization_Logo.svg.png" alt="World Health Organisation logo" title="World Health Organisation logo" width="300" style="float: left;  padding: 10px;">
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/%D7%94%D7%9E%D7%99%D7%9C%D7%94_%D7%A2%D7%91%D7%A8%D7%99%D7%AA_%D7%91%D7%9B%D7%AA%D7%91_%D7%95%D7%91%D7%9B%D7%AA%D7%91_%D7%94%D7%A2%D7%91%D7%A8%D7%99_%D7%94%D7%A7%D7%93%D7%95%D7%9D.jpg/320px-%D7%94%D7%9E%D7%99%D7%9C%D7%94_%D7%A2%D7%91%D7%A8%D7%99%D7%AA_%D7%91%D7%9B%D7%AA%D7%91_%D7%95%D7%91%D7%9B%D7%AA%D7%91_%D7%94%D7%A2%D7%91%D7%A8%D7%99_%D7%94%D7%A7%D7%93%D7%95%D7%9D.jpg" alt="The word HEBREW written in modern Hebrew language and written in Paleo-Hebrew alphabet. Eliran t: CC BY-SA 4.0" title="The word HEBREW written in modern Hebrew language and written in Paleo-Hebrew alphabet. Eliran t: CC BY-SA 4.0" width="200" style="float: left; padding: 10px;">
#
# <p style="clear: left;">
# In practice, you can define your own entities that are relevant to the sort of data you are working with. For example, a corpus of archaeological reports might need additional entities for 'culture', 'material' and 'method' in order to extract useful information within the texts.
#     <p>
#
# Recognising named entities means finding and classifying tokens according to the types of entities you have defined. Named entities can be single tokens, such as 'Berlin' (a place), or they can be spans or contiguous sequences of tokens, like 'The Royal Society' (an organisation) and 'Charles Robert Darwin' (a person).
#
# Once you have a corpus tagged with named entities, you could enrich the data further by linking each entity to a corresponding unique identifier from a knowledge base, such as [VIAF](https://viaf.org/). We will cover this further topic in the [last notebook](5-linking-named-entities.ipynb).

# ---
# ---
#
# ## NER with Machine Learning using spaCy
# [spaCy](https://spacy.io) is a free and open-source package that can be used to perform automated named entity recognition.
#
# If you're interested in learning how to work with spaCy more broadly for a variety of NLP tasks I recommend the beginner's tutorial [Natural Language Processing with spaCy in Python](https://realpython.com/natural-language-processing-spacy-python/) and, for intermediate programmers, the book [Natural Language Processing with Python and spaCy](https://nostarch.com/NLPPython).
#
# spaCy uses a form of machine learning called **convolutional neural networks** (CNNs) for named entity recognition. The nature of these neural networks is out of scope for this workshop, but in general terms a CNN produces a **statistical model** that is used to **predict** what are the most likely named entities in a text. This type of machine learning is described as being **supervised**, which means it must be trained on data that has been correctly labelled with named entities before it can do automated labelling on data it's never seen before.
#
# As a statistical technique, the predictions might be wrong; in fact, they often are, and it's usually necessary to re-train and adjust the model until its predictions are better. We will cover training a model in the following notebooks.
#
# Fortunately, we don't need to start completely from scratch because spaCy provides a range of [pre-trained language models](https://spacy.io/usage/models#languages) for modern languages such as English, Spanish, French and German. For other languages you can train your own and use them with spaCy or re-train one of the existing models.
#
# The English models that spaCy offers have been trained on the [OntoNotes corpus](https://catalog.ldc.upenn.edu/LDC2013T19). These models can predict the following entities 'out of the box':
#
# <table class="_59fbd182"><thead><tr class="_8a68569b"><th class="_2e8d2972">Type</th><th class="_2e8d2972">Description</th></tr></thead><tbody><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">PERSON</code></td><td class="_5c99da9a">People, including fictional.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">NORP</code></td><td class="_5c99da9a">Nationalities or religious or political groups.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">FAC</code></td><td class="_5c99da9a">Buildings, airports, highways, bridges, etc.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">ORG</code></td><td class="_5c99da9a">Companies, agencies, institutions, etc.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">GPE</code></td><td class="_5c99da9a">Countries, cities, states.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">LOC</code></td><td class="_5c99da9a">Non-GPE locations, mountain ranges, bodies of water.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">PRODUCT</code></td><td class="_5c99da9a">Objects, vehicles, foods, etc. (Not services.)</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">EVENT</code></td><td class="_5c99da9a">Named hurricanes, battles, wars, sports events, etc.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">WORK_OF_ART</code></td><td class="_5c99da9a">Titles of books, songs, etc.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">LAW</code></td><td class="_5c99da9a">Named documents made into laws.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">LANGUAGE</code></td><td class="_5c99da9a">Any named language.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">DATE</code></td><td class="_5c99da9a">Absolute or relative dates or periods.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">TIME</code></td><td class="_5c99da9a">Times smaller than a day.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">PERCENT</code></td><td class="_5c99da9a">Percentage, including ”%“.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">MONEY</code></td><td class="_5c99da9a">Monetary values, including unit.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">QUANTITY</code></td><td class="_5c99da9a">Measurements, as of weight or distance.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">ORDINAL</code></td><td class="_5c99da9a">“first”, “second”, etc.</td></tr><tr class="_8a68569b"><td class="_5c99da9a"><code class="_1d7c6046">CARDINAL</code></td><td class="_5c99da9a">Numerals that do not fall under another type.</td></tr></tbody></table>
#
# <p style="text-align: center; font-style: italic;">Table from:
# <a href="https://spacy.io/api/annotation#named-entities" target="_blank">https://spacy.io/api/annotation#named-entities</a>
#     </p>
#
# For more detail about how to use spaCy for NER, see the documentation [Named Entity Recognition 101](https://spacy.io/usage/linguistic-features#named-entities-101).
#
# An alternative to spaCy is [Natural Language Toolkit (NLTK)](https://www.nltk.org/), which was the first open-source Python library for NLP, originally released in 2001. It is still a valuable tool for teaching and research and has better support in the community for older and non-Indo-European languages; but spaCy is easier to use and faster, which is why I'm using it here.

# ---
# ---
#
# ## NER in Practice: A Letter from William Christy, Jr.,  to John Henslow
# Included with these Jupyter notebooks are a set of data from the Henslow Correspondence Project (HCP), which locates and transcribes the letters of the botanist and mineralogist, [John Stevens Henslow](https://www.darwinproject.ac.uk/john-stevens-henslow) (1796–1861). John Henslow was professor of Botany and Mineralogy at the University of Cambridge, and mentor and friend to Charles Darwin; they exchanged significant letters throughout their lives.
#
# <a href="http://resource.nlm.nih.gov/101418379"><img src="https://epsilon.ac.uk/css/images/henslow.jpg" alt="Portrait of J.S. Henslow, by T.H. Maguire, 1851. Half-length, seated, full face; arm resting on books on table; holding spectacles." title="Portrait of J.S. Henslow, by T.H. Maguire, 1851. Half-length, seated, full face; arm resting on books on table; holding spectacles." width="250"></a>
# <p style="text-align: center; font-style: italic;">Portrait of J.S. Henslow, by T.H. Maguire, 1851.</p>
#
# The latest release of letters can be viewed on [Epsilon](https://epsilon.ac.uk/search?sort=date;f1-collection=John%20Henslow), a collaborative platform for reuniting nineteenth-century letters of science.
#
# > **Content warning**: The HCP letters were written during the period of British imperialism, therefore some of the correspondence contains content we now find offensive, for example, `letters_138.xml` contains a racist description. These Jupyter notebooks do not contain or discuss any of this material, but please be aware you may come across it if you browse through the letters independently.
#
# You can browse the letters included with these notebooks by clicking this link to the folder: [`data/henslow`](data/henslow/)
#
# The letters are included here courtesy of HCP and under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
#
# ---
# ---
# ## Inspecting the Letter XML
#
# We start with a letter William Christy, Jr. sent to Henslow in 1831 in which he discusses sending and receiving rare plant specimens, including the rather delightful *Arbutus unedo* (Strawberry tree).
#
# <a title="Arbutus berries (Palombaggia, Corsica). Jplm, Public domain, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Arbouses.jpg"><img width="256" alt="Arbutus berries (Palombaggia, Corsica)" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Arbouses.jpg/256px-Arbouses.jpg"></a>
# <p style="text-align: center; font-style: italic;">Arbutus berries (Palombaggia, Corsica)</p>
#
# Christy was a Fellow of the Linnean Society and a non-resident member of the Botanical Society of Edinburgh.
#
# ---
# > **EXERCISE**: Open the letter now in your browser by clicking this link: [`letters_152.xml`](data/henslow/letters_152.xml).
#
# ---
#
# Your browser should try to display the XML in a relatively friendly form.
# As you scroll down the letter, notice the various types of information that are marked up in the `<teiHeader>`. Collapse or scroll past the `<teiHeader>` to the `<text>` element where the body of the letter itself is marked up. We are interested in the contents of **`<div type="transcription">`**.
#
# Christy begins:
#
# > *I am ashamed that I have never before acknowledged the receipt of the very valuable packet of specimens you were so kind as to send me. Having now got my plants up from Cheshire I shall lose no time in making up a parcel for you.*

# ---
# ---
# ## Opening the Letter with Beautiful Soup
# [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) is a library for getting data out of HTML and XML documents. It makes it easy to search documents for particular information. It's often used for *web scraping*, which involves harvesting data from websites automatically. It can also be used to extract information from plain XML files, which is how we will be using it here.
#
# First we import what we need from the library:

# In[1]:


from bs4 import BeautifulSoup


# Then we open the letter using the `open()` function and pass the file into `BeautifulSoup`. This is jokingly called "making the soup".

# In[6]:


with open("../data/henslow/letters_152.xml", encoding="utf-8") as file:
    letter = BeautifulSoup(file, "lxml-xml")


# The letter has been opened, parsed by `BeautifulSoup` and stored with the name `letter`. Print out the text of the letter below to check the XML is what you expect.

# In[7]:


# Your code here
letter


# When we inspected the XML before, we saw that the actual content of Christy's letter is in the `<div>` element with the attribute `type` that has the value `"transcription"`.
#
# Beautiful Soup allows us to pick out text from a document by element, attribute and/or value. In this case, we will use the **`find()`** method and pass in what we are looking for: an element with the `type` attribute and the value `"transcription"`.
#
# We only want the textual content, not the whole element including the tag, so we access the `text` attribute.
#
# Feel free to experiment and find out what happens if you miss off the attribute `text`.

# In[13]:


transcription = letter.find(type="transcription").text
# transcription = letter.find(type='transcription')

print(transcription)
print(len(transcription))


# Can you isolate text from other elements in the letter instead? Refer to the Beautiful Soup documentation on [Searching the tree](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#searching-the-tree).

# In[ ]:


# Your code here


# ### Cleaning the Transcription
#
# One small point of cleaning we will do now will make a big difference to the quality of NER later on. The style of writing in these letters is to replace the word 'and' with the ampersand (`&`). Experience tells me that spaCy's language model doesn't handle this very well because it is different in style to the texts it was trained on.

# In[14]:


transcription = transcription.replace("& ", "and ")


# In[15]:


transcription


# Ideally, we would instead re-train the language model to better understand the use of ampersands. In this notebook, I just want to show you a good example.
#
# As you follow along, you may notice more aspects of the text that could or should be cleaned. Do make a note of your thoughts.

# ---
# ---
# ## A Pre-Trained Language Model
#
# We can now move into the named entity recognition using spaCy. We are going to use a pre-trained English language model provided by spaCy called `'en_core_web_sm'`. The elements of this model's name break down as follows:
#
# * `'en'`: the language code, in this case, English
# * `'core'`: the capabilities for the model - 'core' is general purpose
# * `'web'`: the type of text the model was trained on
# * `'sm'`: how large the model is when stored on disk - 'sm' means small
#
# A full list of the [available pre-trained language models](https://spacy.io/usage/models#languages) is available.
#
# To start, we import and load the pre-trained English language model and give it the name `nlp`, ready to do the work on the transcription. (The name could be anything, but `nlp` is used by convention, and you will see this name used in examples and tutorials.)

# In[16]:


import en_core_web_sm

nlp = en_core_web_sm.load()


# Before we go any further, let's take a moment to examine the metadata of this language model, and remind ourselves of what it consists. We can do this with the `meta` attribute:

# In[17]:


nlp.meta


# Scroll through this metadata and in particular note the `'description'` and `'sources'`, including information on what data the model is trained on, and `'ner'` labels, which should match those listed near the top of this notebook.
#
# If you have forgotten already what a label stands for (some of them are rather cryptic!) then you can ask spaCy to give you a human-readable explanation:

# In[18]:


import spacy

spacy.explain("FAC")


# Note also the `'accuracy'` statistics, for your interest. The ones relevant to us are:
#
# `'ents_f': 85.5515654809,
#  'ents_p': 85.8937524646,
#  'ents_r': 85.212094115`
#
# These were created when the model was evaluated and give you an idea of how well it performs. `'ents_p'`, `'ents_r'`, `'ents_f'` stand for "precision", "recall" and "fscore", respectively.
#
# **Precision** is the percentage of entities predicted that are actually relevant (_aka_ "positive predictive value"); **recall** is the percentage of relevant entities that were actually predicted (_aka_ "sensitivity"); the **fscore** combines the precision and recall into a single average score.
#
# Put rather simplistically, we can say that this model is about 85% accurate.

# ---
# ---
# ## Recognising Named Entities with spaCy
#
# Now, we can pass the transcription into the language model and spaCy does the rest. It returns to us a `Doc` object (which we name `document`) that contains all the **tokens** and **annotations** it has created. We print out the document text stored on the attribute `text` just to check that spaCy has correctly parsed the transcription.

# In[19]:


document = nlp(transcription)
document.text


# But what has actually happened? Behind the scenes spaCy has done the following tasks:
#
# * tokenizing and lemmatizing
# * part-of-speech tagging ('tagger')
# * syntactic dependency parsing ('parser')
# * named entity recognition ('ner')
#
# This series of tasks is called a **processing pipeline** and each individual task is known as a **component**.
#
# <img src="https://spacy.io/pipeline-7a14d4edd18f3edfee8f34393bff2992.svg" alt="Default spaCy processing pipeline" title="Default spaCy processing pipeline">
#
# Of all these components (tasks), the tokenizer is essential and must happen for every pipeline; the rest are optional, depending on what you want to do with your text.
#
# In the metadata for this model (`nlp.meta` above), you can see listed the pipeline components:
#
# `'pipeline': ['tagger', 'parser', 'ner']`
#
# This is the default processing pipeline included with the `'en_core_web_sm'` language model. You can read more about [spaCy Processing Pipelines](https://spacy.io/usage/processing-pipelines).
#
# As we only need the `'ner'` component, we could speed up our code by disabling the tagger and the parser like this:
#
# `nlp = en_core_web_sm.load(disable=['tagger', 'parser'])`

# In[ ]:


# Write your code here to compare the performance of the model with and without the unnecessary components
# Hint: you could use the Jupyter magic command `%%time` or `import time` and use `time.time()`


# Now, we just need to inspect the data. All the named entities are available on the `ents` attribute. The following code loops over all the named entities and prints each of them out with the labels that spaCy has predicted.

# In[20]:


for entity in document.ents:
    print(f"{entity.text}: {entity.label_}")


# What do you think of the accuracy of the labelling?

# ### Exporting Named Entities in JSON Format
# We can export these named entities in a JSON format, which can be saved in a file or used as input to other spaCy functions.

# In[21]:


# Create a dictionary form of the document data
doc_dict = document.to_json()
doc_dict


# In[22]:


# Get just the named entities item
ents_dict = {key: value for (key, value) in doc_dict.items() if key == "ents"}
ents_dict


# In[23]:


# Import a module for handling json
import json

# Export the named entities dictionary in json format
json.dumps(ents_dict)


# ---
# ---
# ## Visualising Named Entities with displaCy
#
# The list above could be useful, but it is much easier as a human to understand how accurate the NER has been if we can see where the named entities fall within the context of the document.
#
# Fortunately, spaCy provides a nice visualiser called **displacy** that does this for us.

# In[24]:


from spacy import displacy

displacy.render(document, style="ent")


# ---
# > **EXERCISE**: Inspect the visualisation above and make a note of what spaCy has predicted well and what it is has got wrong. Has it mislabelled some tokens? Has it missed some entities out? Has it got some of the spans wrong?
#
# ---
#
#

# We can also **save** the results of a visualisation as an HTML file to review later:

# In[25]:


# Import a module that helps with filepaths
from pathlib import Path

# Create a filepath for the output file
output_file = Path("../results/ent_viz.html")  # Path("output/ent_viz.html")

# Give the document a title for reference
document.user_data[
    "title"
] = "Letter from William Christy, Jr., to John Henslow, 26 February 1831"

# Output the visualisation as HTML
html = displacy.render(document, style="ent", jupyter=False, page=True)

# Write the HTML to the output file
output_file.open("w", encoding="utf-8").write(html)


# Now navigate to the file in the Jupyter file listing, or click on this link to view the file: [`output/ent_viz.html`](output/ent_viz.html). (Note that this file will not exist until you have run the code above!)
#
# The makers of spaCy also run a [named entity visualizer demo](https://explosion.ai/demos/displacy-ent) on their website, where you can experiment with different texts without writing any code.

# ---
# > **EXERCISE**: Go back to the code where we parsed the XML of the letter with Beautiful Soup and try running the NER on some other letters. Remember you need to run all the following cells in order.
#
# ---

# ---
# ---
# ## Summary
#
# In this notebook:
#
# - A **named entity** is defined as any type of real-world object or concept that is assigned a name or proper name.
# - The Python library **BeautifulSoup** can parse text in XML, which is useful for extracting the text of Henslow letters marked up in TEI.
# - The Python library **spaCy** can predict named entities by using an English language model that has been pre-trained using machine learning on a corpus of general modern texts.
# - spaCy can recognise certain **pre-defined** named entities 'out of the box'.
# - The **predictions** made by spaCy's default model are quite good but suffer from some inaccuracies.
# - spaCy can **visualise** the named entity labels within the context of the original text so we can better assess the accuracy of the predictions.
#
# In the [next notebook](3-principles-of-machine-learning-for-named-entities.ipynb) we will look in more detail at how machine learning **models** make **predictions** about named entities, how to improve the predictions by **training** the model and how to create **training data** by labelling data manually.
