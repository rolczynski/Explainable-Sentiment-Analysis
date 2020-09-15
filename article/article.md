
** Explainable Sentiment Analysis **
#open-source #NLP #deep-learning #aspect-based-sentiment-classification



introduction

<br>

#### Problem Definition

The aim is to classify the sentiment of the text for several aspects.
We have made several assumptions to make the service more reliable and useful.
The processing text might be long, for instance, a full-length document.
The aspect can contain several words so it may be defined more precisely.
Moreover, the user is able to infer how much results are reliable
because the service provides an approximate decision explanation.
We will discuss it closely later on.

```python
import aspect_based_sentiment_analysis as absa

nlp = absa.load()
text = ("We are great fans of Slack, but we wish the subscriptions "
        "were more accessible to small startups.")

slack, price = nlp(text, aspects=['slack computer program', 'price'])
assert price.sentiment == absa.Sentiment.negative
assert slack.sentiment == absa.Sentiment.positive
```

Above is an example how quickly you can start to benefit from our open-source package.
The `load` function sets up the ready-to-use pipeline `nlp`.
You can explicit pass a model name you wish to use (the list of available models is [here],
or a path to your model.
We encourage you to build your own model which reflects your data.
The results will be more accurate and stable.

<br>


#### Pipeline: Keep Process in Shape

The pipeline provides an easy-to-use interface to make predictions. 
The highly accurate model is useless if it is unclear how to correctly prepare the inputs
and how to interpret the outputs. Therefore, to make things clear, we introduce the
pipeline that is highly connected with the model type. In the previous example, we present
how to start to work with a default pipeline. Nonetheless, to build both your own model
and pipeline, it is worth to know how we face the problem in detail.

```
pre-process -> predict -> review -> post-process
```

The diagram above illustrates an overview how the pipeline works.
At the very beginning, we convert the input, the text and aspects, into the `task`.
The task keeps the well-prepared tokenized examples that we further encode and pass to the model.
The fine-tuned model makes a prediction, and, instead of directly post-processing model outputs 
and return results, we add one additional step which is optional.
We introduce the review process wherein the `professor` verifies model predictions and reasoning
to make the final decision.
It might dismiss a model prediction if either the model reasoning or outputs seem to be suspicious.
In the next sections, we will discuss how the model and professor work in detail.

````python
import aspect_based_sentiment_analysis as absa

name = 'absa/bert_abs_classifier-rest-0.1'
model = absa.BertABSClassifier.from_pretrained(name)
tokenizer = absa.BertTokenizer.from_pretrained(name)
professor = absa.Professor(...)     # Explained in detail later on.
text_splitter = absa.sentencizer()  # The English CNN model from SpaCy.
nlp = absa.Pipeline(model, tokenizer, professor, text_splitter)

# Break down the pipeline `call` method.
task = nlp.preprocess(text=..., aspects=...)
input_batch = nlp.encode(task.tokenized_examples)
output_batch = nlp.predict(input_batch)
predictions = professor.make_decision(task, output_batch)
completed_task = nlp.postprocess(task, output_batch, predictions)
````

Above is an example how to initialize the pipeline directly,
and, to revise in code the process mentioned before, 
we expose explicitly what calling the pipeline does.
We try to omit many details of little account, 
nonetheless, one thing we wish to highlight.
The sentiment of a long text tends to be fuzzy and neutral. 
Therefore, you might split a document text into smaller independent chunks, spans. 
They can include a single sentence or several sentences.
It depends how works the `text_splitter`. 
In this case, we use the SpaCy CNN model which 
splits a document into single sentences, and, in consequence,
each sentence is processing independently.
Note that longer spans have richer context information, 
so a model has more insights to make a decision.
Please take a look at the pipeline details [here].

<br>


#### Model: Heart of Pipeline

The model's aim is to classify the sentiment of the text for the given aspect.
This is the challenging task because the sentiment is frequently meticulously hidden.
Nonetheless, before we jump into our model details, 
we look closely how researchers approach the task in the past,
what is the language model, and why it makes a difference.

The model is a function that maps the input to the desired output.
Because this function is unknown, we try to approximate it using data.
It is hard to build the clean and big enough, for NLP tasks in particular,
dataset to train model directly in a supervised manner.
Therefore, the basic approach to solve this problem, 
and overcome the lack of a sufficient amount of data, 
is to construct hand-crafted features which we treat as part of the model.
Engineers extract key tokens, phrases, n-grams, and train a classifier 
to assign weights how likely these features are either positive or negative.
Based on these human-defined features, a model makes predictions.
This is a valid approach, popular in the past, 
but such simple models could not precisely capture the complex natural language,
and, in consequence, they reached the accurateness limit not to overcome.

```
feature engineering vs. lanugage model
```

Researchers have made a breakthrough when they started to transfer knowledge 
from language models to down-stream, more specific, NLP tasks.
Nowadays, slowly it becomes the standard that the key component of the modern NLP system is the language model.
Briefly, the language model serves rough natural language understanding.
In the separate heavy in computation training, it processes enormous datasets, 
like the entire Wikipedia or more, to figure out relations between words.
As a result, it is able to encode words, meaningless strings, into rich in information vectors.
Because encoded-words, context-aware embeddings, 
live in the same continuous space, we can effortlessly manipulate them.
If you wish to, for instance, summarize a text, you might sum vectors; 
compare two words, make a dot product between them, etc.
Rather than using feature engineering, and engineers' linguistic expertise (the implicit knowledge transfer),
we benefit from the language model as the ready-to-use, portable, and powerful features provider.

With this context, we are ready to define our model, both powerful and simple.
The model contains the language model `bert`, which provides features, and the linear `classifier`.
Among variety of language models, we use BERT,
because we can benefit directly from the BERT's next-sentence prediction, 
and formulate the task as the sequence-pair classification. 
In consequence, an example is described as one sequence in the form:
"[CLS] text subtokens [SEP] aspect subtokens [SEP]". 
The relation between a text and an aspect is encoded into the [CLS] token. 
The classifier just makes a linear transformation of the final special [CLS] token representation.

````python
import transformers
import aspect_based_sentiment_analysis as absa

name = 'absa/bert_abs_classifier-rest-0.1'
model = absa.BertABSClassifier.from_pretrained(name)
# The model has two essential components:
#   model.bert: transformers.TFBertModel
#   model.classifier: tf.layers.Dense 

# We've implemented the BertABSClassifier to make room for further research.
# Nonetheless, even without installing the `absa` package, you can load our 
# bert-based model directly from the `transformers` package.
model = transformers.TFBertForSequenceClassification.from_pretrained(name)
````

Even if it is out of the scope of this blog post, note how we train the model from scratch. 
We start with the original BERT version as a basis, and we divide the training into two stages. 
Firstly, due to the fact that the BERT is pretrained on dry Wikipedia texts, 
we bias the language model towards the more informal language or a specific domain. 
To do so, we select raw texts close to the target domain and do the self-supervised
**language model** post-training. 
The routine is the same as for the pre-training but we need carefully set up optimization parameters.
Secondly, we do the regular supervised training. 
We train the whole model jointly, the language model and the classifier, 
using the fine-grained labeled dataset.

<br>


#### Awareness of Model Limitations

In the previous section, we have presented an example of the modern NLP architecture 
wherein the core component is the language model.
We abandon doing the task specific and time consuming feature engineering.
Instead, thanks to language models, we use powerful features, context-aware word embeddings.
Nowadays, transfer knowledge from language models has become extremely popular.
This approach dominates leader boards of any NLP task 
including the aspect-based sentiment classification (the table below).
Nonetheless, we should interpret these results with caution.

```
The state of the art results on the most common evaluation dataset 
(SemEval 2014 Task 4 SubTask 2, details [here](http://alt.qcri.org/semeval2014/task4/)).
The second model in the exact form has been presented in the previous section.
All use BERT as the language model.
```

The single metric might be misleading, especially,
when an evaluation dataset is highly limited, as in this case.
Briefly, a model seeks any correlations useful to make a correct prediction,
no matter if they have sense for humans or not.
In consequence, the model and human reasoning are much different.
The model encodes the dataset specifics invisible for humans.
In return for the high accurateness of the massive modern models, 
we have a little control of the model behavior 
because both model reasoning and dataset characteristics, 
which a model tries to map precisely, are unclear. 
These drawbacks become a severe problem 
if a dataset is limited because it is more likely during an inference 
to unconsciously expose a model to completely unusual, 
from a model perspective, examples, which cause unpredictable model behavior.
To avoid such dangerous situations 
and better understand what is beyond model comprehension, 
we should construct additional tests, the fine-grained evaluations [survey].


In the table below, we present three basic tests 
that roughly estimate model limitations.
To be consistent, we examine the BERT-ADA model introduced in a previous section.

The test A checks how crucial to predict correctly is information about an aspect.
We restrict a model to predict without aspects.
The task becomes a basic, aspect independent, sentiment classification.

The test B verifies how a model considers an aspect. 
We force a model to make predictions using unrelated aspects, 
manually selected and verified simple nouns 
that are not present in a dataset, even implicitly. 
We process positive and negative examples 
expecting to get neutral sentiment predictions.

The test C examines how precisely a model separates information 
about an asked aspect from other aspects.
In the first case, we process positive and neutral examples
wherein we add to texts a negative emotional sentence
about a different aspect
"The {other aspect} is really bad."
expecting that a model persists predictions unchangeable.
Accordingly, in the second case, we process negative and neutral examples
with an additional positive sentence "The {other aspect} is really great.".
To make this test reliable, we use unrelated aspects from the test B.

```
Test Name                       | Acc. Laptop | Acc. Restaurant
---------------------------------------------------------------
SemEval                         | 80.23       | 87.89
---------------------------------------------------------------
Test A: Course-Grained          | 76.64       |
Test B: Aspect as Constrain     | 10          |
Test C: Add Emotional Sentence  |             | 
        a) negative             | 78 -> 73 (-6.4%)  |
        b) positive             | 74 -> 61 (-17.5%) |

The test results of the BERT-ADA models.  
The B and C tests were run 7 times for different unrelated aspects
so we present means and standard deviations.
```

The test A confirms that the course-grained classifier achieves good results as well.
Even if the aspect-based classifier performs better, the improvement is below 4%.
This is not the good news.
If a model predicts a sentiment correctly without taking into account an aspect,
roughly, there is no gradient towards patterns that support the aspect-based classification,
and a model does have nothing to improve in this direction.
Consequently, at least 76% of the already limited dataset does not help 
to improve the aspect-based sentiment classification.
Besides, these examples might even disturb,
because they overwhelm examples which require multi-aspect consideration.
Multi-aspect examples might be treated as outliers, 
and gentle wiped off, for instance, due to averaging a gradient in a batch.

The test B clearly demonstrates that a model does not truly solve the aspect-based condition, 
because it considers an aspect mainly as a feature, not as a constraint. 
In 10% cases, a model recognizes correctly that text does not concern a given aspect, 
and returns neutral sentiment instead of positive or negative. 
The model's usefulness in terms of aspect-based classification is questionable. 
One might infer this conclusion directly from the model architecture
because there is no dedicated mechanism that might impose such constraint.

The test C supports that a model well separates information about different aspects. 
In the above 80% cases in each test, 
a model correctly deals with a basic multi-aspect problem, 
and recognizes that an added, highly emotional in opposing direction, 
sentence does not concern a given aspect. 
Even if the test B shows that a model neglects information about an unrelated aspect, 
this test reveals that an aspect is a vital feature indeed 
if a text concerns many aspects including a given aspect, 
and a model needs to separate information about different aspects.

The precisely designed tests can provide many valuable insights [checklist].
Unfortunately, papers describing state-of-the-art models gentle leave aside tests
that may expose model limitations.
In contrast to this, we encourage you to do many tests.
Even if they might be time-consuming and tedious, 
the test-driven model development is powerful because
you can understand model defects in detail, fix them, and smoothly reach further improvements.

<br>


#### Professor: Supervise Model Predictions

The tests reveal severe model limitations.
The natural way to eliminate them is fixing a model itself.
Firstly, keeping in mind that a model reflects data,
we can augment a dataset, and add examples that are underrepresented,
in other words, examples that a model has difficulties in predicting correctly.
Since manual labeling is time-consuming, we might think about generating examples.
Nonetheless, models quickly adapt to fixed structures, so benefits might be minimal.
Secondly, of course, we can propose new model architecture.
This is an appealing direction from a researcher perspective,
however, profits from findings are uncertain.

```
pre-process -> the model predicts -> **the professor reviews** -> post-process
```

Due to problematic model fixing, unknown model reasoning, and limited decision control, 
we abandon using the standard pipeline that relies exclusively on the single end-to-end model. 
Instead, before returning the final decision, we fortify the pipeline with the reviewing process. 
We introduce the distinct component `professor` that manages the reviewing process. 
The professor reviews model hidden states and outputs to both identify and correct suspicious predictions. 
It is composed of auxiliary either simple or complex models 
that examine model reasoning and correct model weaknesses. 
Because the professor considers information from aux. models to make the final decision,
we have greater control over decision-making, and we can freely customize model behavior.

````
professor => fix model limitations & explain model reasoning
````

<br>

##### Professor: Fix Model Limitations

Firstly, we highlight how to smoothly start fixing model limitations using the professor.
Coming back to the stated in the test B problem, to avoid questionable predictions,
we wish to build an aux. classifier that predicts whether a text relates to an aspect or not.
If there is no reference in a text to an aspect, the professor sets the neutral sentiment
regardless of a model prediction (details [here]).

````python
import aspect_based_sentiment_analysis as absa

name = 'absa/basic_reference_recognizer-0.1'
recognizer = absa.aux_models.BasicReferenceRecognizer.from_pretrained(name)
professor = absa.Professor(reference_recognizer=recognizer)
````

The good practice is gradually increasing model complexity in response to demands.
Because the reference recognition is a side problem right now, 
we simplify the task and propose the aux. model `BasicReferenceRecognizer` that only checks 
if an aspect is clearly mentioned in a text
(details [here]).
Even if the table below confirms an improvement, note that this is a simple test case.
We encourage you, especially if it concerns your business, to construct more challenging tests
wherein aspect mentions are more implicit.

```
Name                            | Acc. Laptop | Acc. Restaurant
---------------------------------------------------------------
Test B: Aspect as Constrain     | 10          | 11
---------------------------------------------------------------
Test B: HotFix                  | 94          | 87

The tests were run 7 times for different unrelated aspects
so we present means and standard deviations.
We set up the cosine similarity threshold 0.1 based on the training data.
```

<br>

##### Professor: Explain Model Reasoning

As we said, there is the second equally important (perhaps primary) reason why we introduce the professor.
The professor aims to explain model reasoning what is extremely hard.
We are far from explaining model behaviour precisely
even though it is crucial for building intuition that fuels further research and development.
Besides, model transparency enables understanding model failures from various perspectives, 
considering safety (e.g. adversarial attacks), fairness (e.g. model biases), reliability (e.g. spurious correlations), and more.

```
understand model reasoning => development - safety - fairness - reliability
```

Explaining model decisions is challenging not only due to model complexity.
As we said before, models make decisions in a completely different way than people do.
We need to translate abstract model reasoning into a human understandable form.
To do so, we break down model reasoning into components that we can understand, patterns.
The pattern is interpretable and has the `importance` attribute (within the range <0, 1>) 
that expresses how a particular pattern contributes to the model prediction.

The single `pattern` is a weighted composition of input tokens.
It is a vector that assigns for each token a weight (within the range <0, 1>)
that defines how a token relates to a pattern.
For instance, a one-hot vector illustrate a simple pattern that is composed of a single token.
This interface, the pattern definition, enables to convey either simple or more complex relations.
It can capture rationales (a binary vector that defines a subset of input tokens), 
key tokens (one-hot vectors), or more tangled structures.
The more complex interface, the pattern structure, more details can be encoded.
In the future, the interface would be the natural language itself.
It would be great to read a decision explanation in the form of an essay.

```
the subset of tokens (latent rationales) - the ranking of single tokens - complex structures
----- pattern complexity ----->
```

The key idea is to frame the problem of explaining a model decision as an independent task wherein
an aux. model, the `pattern recognizer`, predicts patterns given model inputs, outputs, and internal states.
This is a flexible definition so we will be able to test various recognizers in a long perspective.
We can try to build model-agnostic pattern recognizer 
(independent with respect to the model architecture or parameters).
We can customize inputs, for instance, take into account internal states or not,
analyze a model holistically or derive conclusions only from a specific component.
Finally, we can customize outputs, defining sufficient pattern complexity.
Note that it is challenging to design the training and evaluation because true patterns are unknown.
In consequence, extracting complex patterns correctly is extremely hard.
Nonetheless, there are few successful attempts to train a pattern recognizer that reveals latent rationales.
This is a case in which a pattern recognizer tries to mask-out as many input tokens as possible constrained to 
keeping an original prediction (e.g. the `DiffMask` method [here], and perturbation-based methods [here]).

<p align="middle">
<img src="images/patter-recognizer.svg" width="600" alt=""/>
</p>

<br>

#### Dive into Pattern Recognizer

Due to time constraints, we did not want to build a trainable pattern recognizer at first.
Instead, we decided to start with a pattern recognizer that comes from our observations, the prior knowledge.
The model, the aspect-based sentiment classifier, is based on the transformer architecture [here] 
wherein self-attention layers hold most parameters, therefore,
one might conclude that understanding self-attention layers is a good proxy to understand a model as a whole.
Accordingly, there are many articles that manifest how to explain a model decision 
in simple terms using attentions (internal states of self-attention layers) straightforwardly [here].
Inspired by these articles, we also analyze attentions (processing training examples) looking for meaningful insights.
This exploratory study leads us to form the `BasicPatternRecognizer`
(details in an appendix below and implementation [here]).

````python
import aspect_based_sentiment_analysis as absa

# This pattern recognizer doesn't have trainable parameters
# so we can initialize it directly, without setting weights. 
recognizer = absa.aux_models.BasicPatternRecognizer()   
professor = absa.Professor(pattern_recognizer=recognizer)

# Examine the model decision.
nlp = absa.Pipeline(..., professor=professor) # Set up the pipeline. 
slack = nlp(text=..., aspects=['slack computer program'])
absa.display(slack.review) # We use IPython so it works inside a notebook.
````

<p align="middle">
<img src="images/patterns.gif" width="600" alt=""/>
</p>

<br>

Forming the basic pattern recognizer, we have made severe assumptions
so we should be careful about interpreting explanations literally.
Even if attention values have thought-provoking properties, for instance, they encode rich linguistic relations [here],
there is no proven chain of causation.
There are many articles that manifest various concerns why drawing conclusions about model reasoning
directly from attentions might be misleading [here].
Even if patters seem to be reasonable, the critical thinking is a key.
We need the quantitative analysis to truly measure how correct explanations are.
Like the training, the evaluation of a pattern recognizer is tough due to the fact that true patterns are unknown.
In consequence, we are forced to use proxies that validate only selected properties of predicted explanations.

<br>

##### Pattern Recognizer: Key Token Recognition Test

There are several properties needed to keep an explanation consistent.
For instance, the core token from the most important pattern should impact a model decision significantly.
To confirm whether a pattern recognizer supports this intuitive property or not, we do a simple test. 

```
[Pattern Recognizer] Patterns -> Most Important Pattern -> Key Token
                                       [ ... ]              {fans}
```

We mask in the input sequence directly the chosen based on an explanation token 
(e.g. fans), and observe if a model changes a decision or not.
Note that an example might contain several `key tokens`, 
tokens that masked (independently) cause a change of a model prediction.
We assume that the chosen token should belong to the group of key tokens if it is truly significant.
This test is attractive because we are able to reveal the ground-truth (key tokens),
solely checking `n` combinations (masking each token), needed to measure the test performance precisely.

```
Pattern Recognizers      | Acc. Laptop | Acc. Restaurant
--------------------------------------------------------
Random                   | 10          |
Attention                | 31          |
Gradient                 | 15          |
--------------------------------------------------------
Basic                    | 52          |

The key token recognition based on an explanation provided by a pattern recognizer. 
Evaluated on test examples that have at least one key token (34%).
```

In the table above, we compare four exemplary pattern recognizers
(look at _Pattern Recognizers in detail_, an appendix below, implementation [here]).
To sum up, the 17% of test examples have at least one key token (others we filter out).
Among them, in 52% cases, the chosen token based on an explanation from the basic pattern recognizer is the key token.
From this perspective, the basic patter recognizer is far more precise that other methods
(details and implementation [here]).

<br>

##### Pattern Recognizer: Key Pair Recognition Test

The truth is that we cannot assess an explanation precisely taking into account only a single token.
Keeping this article concise, we cover solely one test more but there are endless properties to review. 
The purpose of this test is to check whether a pair of two core tokens from the most important pattern 
convey the information about the essential (in terms of decision-making) relation between tokens.

```
[Pattern Recognizer] Patterns -> Most Important Pattern ->   Key Pair
                                       [ ... ]             {fans, slack}
```

We do a similar recognition test but now the aim is to predict the `key pair of tokens`,
a pair that masked causes a change of a model prediction.
In contrast to the previous test, now it is much harder to check all pair combinations
to retrieve the ground-truth needed to measure the test performance.
Usually it is practically impossible to do so, and this is the unfortunate implication of unknown model reasoning.
We can compare pattern recognizer results but we cannot say exactly how accurate they are (in most tests).

<p align="middle">
<img src="images/confusion_matrix.svg" width="600" alt=""/>
</p>

<br>

Above, we check correctness of the basic pattern recognizer against other methods.
We benefit from the previous test to withdraw examples (17%) that have at least one key token.
Therefore, we are sure that a decision flip is caused by a key pair, not one token from a pair.
In matrices, on-diagonal values illustrate cases wherein recognizers behave similarly.
In these cases, both recognizers choose a pair that masked 
either flips a model decision (the bottom-right cell) or does not (the upper-left cell).
Off-diagonal values are more revealing because they expose differences.
The bottom-left cell counts examples wherein
the basic recognizer uncovers a key pair correctly while another recognizer does not,
and the other way around (the upper-right cell).
The upper-right value is helpful also to estimate how precise (at most) is the basic recognizer.
To sum up, the basic recognizer aims to maximize the bottom-left and minimize the upper-right rate.
From this perspective as well, the basic patter recognizer stands out from other methods
(details and implementation [here]).

<br>

##### Pattern Recognizer: Complexity of Model Reasoning

Forming either new tests or recognizers is not the goal on its own (at least, in this article).
The aim is to benefit from a pattern recognizer to discover insights 
about a model (and a dataset), and use them to make improvements.
The basic information that we can infer from a pattern recognizer is the complexity of the model reasoning.
We wish to know, for instance, whether a single token usually triggers off a model 
or a model rather uses more sophisticated structures.
Assuming that more complex structures engage more tokens (e.g. more relations potentially),
we approximate the complexity of the model reasoning by the number of tokens crucial to make a decision.
We assume that crucial tokens (decision rationales) are the minimal `key set of tokens`, 
the minimal set of tokens that masked (altogether) cause a change of a model prediction.
As a result, we estimate the complexity using key sets that provide implicitly a patter recognizer.

```
[Model]-[Patter Recognizer] -> Patterns -> [Rule] -> Key Set
```

It is important that the key set comes from a pattern recognizer indirectly.
We need to find a way how to transform patterns into a key set.
The simple approach is to form a basic rule that based on patterns predicts a key set.
In the last two tests, reviewing two basic properties of patterns, we've already set up two specific rules.
We assume that the most important pattern should have 
either the most important token (the key token) or the pair of tokens (the key pair).
Now, we formulate the general rule (details [here]) that summarizes all patterns,
and proposes the key set of a given size `n` (not only 1 or 2).

```python
import aspect_based_sentiment_analysis as absa

patterns = ...
key_set_prediction = absa.aux_models.get_key_set(patterns, n=3)
```

The rule provides solely the key set prediction of a given size.
Therefore, we propose a simple strategy wherein we iterate through key set predictions
aiming to find out the minimal key set, the crucial tokens.
Starting with the `n=1`, we check the correctness of a prediction 
calling a model with the masked tokens that belong to the predicted key set.
If a model decision is changed, it is the final prediction of the key set.
If not, we take the next `n+1` key set prediction, and repeat the process (or break).
Note that the key set is correct (it flips a decision) but there is no guarantee that the key set is minimal.

```
Patterns -> Key Set Prediction -> Check -> Key Set
                               <- `n+1`
```

The chart below presents the summary of the model reasoning in terms of the complexity.
As we said, the complexity is approximated by the number of crucial tokens (the minimal key set)
that we infer from patterns using the rule and strategy.
In the first case, the key set contains one token that is simply the key token.
Therefore, the values for the argument one reflects exactly the key token (first) recognition test.
For instance in laptops, the basic pattern recognizer predicts correctly 52% of time in 17% of total examples.
In the next case, the key set contains two tokens, the key pair.
In contrast to the key pair (second) test wherein before a prediction we filter-out examples that have a key token,
now, the pattern recognizer works independently, without the prior knowledge.
The values (on the plot and in the test) might differ because there is no guarantee that the inferred key set is minimal,
and, in this case, some key pairs might cover overlooked key tokens.
To have the broader context, we retrieve and show up the ground-truth of the key pair test as well (details [here]).

```
Tokens 1... 5+ 
- the ground-truth / first two columns (green) 
- the base recognizer (light gray with contours)

The model reasoning complexity within test datasets (laptops and restaurants)
approximated by the probability mass function of the number of tokens in the minimal key sets.
```

The model usually uses few tokens to make a decision, it rarely triggers off on single tokens.
The decision in more than half cases is based on simple patterns (one or two tokens) but
the significant number of examples belongs to last `5+` group what seems suspicious.
Reviewing this group we quickly note that
the model returns a positive sentiment for the almost completely masked input sequences.
In an appendix, to clearly observe this limitation, 
we compare the complexity of model reasoning separating the positive and negative examples.

<br>

The ground-truth is helpful but not crucial.
The true distribution is unknown but it is right-side bounded by the predicted distribution.
The more precise recognizer would try to push the predicted distribution towards left-hand side,
therefore, without the ground-truth, still we can reveal impactful insights about a model.
There are more valuable insights if patterns and inferred key sets are more precise. 
In an appendix, we compare accurateness of different pattern recognizers,
and the basic pattern recognizer clearly outperforms other methods.

<br>

In this section, we present an exemplary use case how to infer from patterns.
Apart from different use cases, there are likely other ways to estimate 
the complexity of the model reasoning, not only using key sets.
However, independently to the use case and approach,
there is another important factor of inferring from patterns, the number of additional calls to a model.
In this case, instead of using the plain rule, one can build another model that based on patterns predicts 
whether an example has a key token or the key set of four tokens without any additional calls to a model.
As a result, the model interpretation does not interfere with the model inference efficiency.

<br>

#### Rolf: Exemplary Graphical User Interface

My dear colleagues have built [https://rolf.scalac.io](https://rolf.scalac.io), 
an exemplary GUI that presents the package capabilities.
Without installing the package, you have a quick access to the default pipeline that the `absa.load` function returns.
The backend solely wraps up a pipeline into the clean flask service which provides data to the frontend.
Write your text, add aspects (optionally), and hit the button "analyze".
In a few seconds, you have a response, the explained prediction.

<p align="middle">
<img src="images/rolf-main-page.png" width="600" alt=""/>
</p>

On the right-hand side, for given aspects, there is the overall sentiment,
and below, the sentiment within independent spans (in this case, sentences that come from the `sentenzier`).
Click on a span, and the inspect mode window pops up.
This is the visualization of the professor review, namely, 
the information whether a span refers to an aspect or not, and the decision explanation.
Review, clicking on different dots, the most significant patterns
that come directly from the basic pattern recognizer, and explain a model decision. 

<p align="middle">
<img src="images/rolf-inspect-mode.png" width="600" alt=""/>
</p>

Please be aware that the service is running on the CPU "minimal-resource" machine,
therefore, the inference time is extremely high comparing to the well-adjusted modern computational units.
The adjustment is straightforward having the specification about constraints
and the data stream that the service aims to process.

<br>
