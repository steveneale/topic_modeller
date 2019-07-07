# topic_modeller

*topic_modeller* is a program for training LDA (Latent Dirichlet Allocation)-based topic models, implemented using
[Python](https://www.python.org/) and [scikit-learn](https://scikit-learn.org/stable/).


## Dependencies

*topic_modeller* is written in [Python](https://www.python.org/), and so a recent version of *Python 3* should be
downloaded before using it. Downloads for *Python* can be found at https://www.python.org/downloads/.

*topic_modeller* depends on a number of external libraries, and so a `requirements.txt` file has been included in
the root directory. To run it from the command line, type:

```bash
pip install -r requirements.txt
```

## Usage

*topic_modeller*'s entry point is the `TopicModeller` class, which can be imported and used either using the Python
interpreter or as part of your own Python project.


### Training a topic model

To train a topic model, instantiate a new instance of the `TopicModeller` class and call
it's `build_topic_model` function, passing the following arguments:

* input file path - .csv file containing training data to be processed.
* dataset (*keyword arg*) - the type of dataset being loaded (`"abcnews"` [default]).

For example

```python3
from topic_modeller import TopicModeller

modeller = TopicModeller()
modeller.build_topic_model("relative/path/to/input.csv", dataset="abcnews")
```

Once training has successfully completed, a new `TopicModel` object will be created containing the trained LDA model
and count vectoriser created during training, and will be assigned to the `TopicModeller` instance's `topic_model`
attribute.


### Saving a trained topic model

To save a `TopicModeller` instance's trained `topic_model` (`TopicModel` object), call
`TopicModeller`'s `save_topic_model_with_name` function, passing the following arguments:

* output model name - name with which to save the trained topic model.

For example:

```python3
modeller.save_topic_model("new_model")
```

A new directory `output/models/new_model/` will be created, containing `topic_model.pkl` and `vectoriser.pkl`.
