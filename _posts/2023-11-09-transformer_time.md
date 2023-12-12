---
layout: distill
title: Transformer for long sequence time series data
description: A project for MIT 2023 Fall 6.S898 Deep Learning about the adaptation of transformer in long sequance time series data, in the case of traffic prediction.
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Jie Fan
    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: MIT
  # - name: Boris Podolsky
  #   url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
  #   affiliations:
  #     name: IAS, Princeton
  # - name: Nathan Rosen
  #   url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
  #   affiliations:
  #     name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography:  2023-11-09-transformer_time.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Abstract
  # - name: Images and Figures
  #   subsections:
  #   - name: Interactive Figures
  - name: Literature Review
  - name: Research Question
  - name: Problem Formulation
  - name: Experiments
  - name: Conclusion and futrue work

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Abstract
This research means to discover the power of transformer in dealing with time series data, for instance traffic flow. Transformer with multihead self-attention mechanism is well-suited for the task like traffic prediction as it can weight the importance of various aspects in the traffic data sequence, capturing both long-term dependencies and short-term patterns. Compared to the LSTM, the transformer owns the power of parallelization, which is more efficient when facing a large dataset. And it can capture the dependencies better with long sequences. However, the transformer may have trouble dealing with the long sequence time-series data due to the heavy computation. This research try to compare differnt methods and their combination from the perspective of computational efficiency and prediction accuracy. 

## Introduction

The time series data processing and prediction are usually conducted with RNN and LSTM. In the case of traffic prediction, CNN and GNN are combined for efficiently capturing spatial and temporal information. And LSTM is widely used as its better performance on capturing temporal dependencies. While recent studies have propsed to replace RNNs with Transformer architecture as it is more efficient and able to capture sequantial dependencies. However, the model is inapplicable when facing long sequence time-series data due to quadratic time complexity, high memory usage, and inherent limitation of the encoder-decoder architecture. <d-cite key="Zhou_Zhang_Peng_Zhang_Li_Xiong_Zhang_2021"></d-cite> 

MAE has been applied to various types of data and problems, including text(BERT)<d-cite key="devlin2019bert"></d-cite>  and image<d-cite key="He_2022_CVPR"></d-cite>. However, it has not been widely discussed on traffic data. The masking ratio in MAE is related to the information redundancy of the data. Natural images are more information-redundant than languages and thus the optimal masking ratio is higher. BERT<d-cite key="devlin2019bert"></d-cite> uses a masking ratio of 15\% for language, MAE<d-cite key="He_2022_CVPR"></d-cite> uses 75\% for image and the optimal ratio for video is up to 90\%.<d-cite key="feichtenhofer2022masked"></d-cite> Traffic data is potentially redundant. It contains temporal and spatial information so that neighbor sensors can provide extra information in addition to temporal consistency. We inducted that the optimal ratio for traffic data should be located between image and video. As it has multidimensional information than image and the speed captured by sensors is not as consistent as the frames in videos.

## Methodology
The information redundancy leads to the common solutions of using transformer to deal with long sequence time-series forecasting(LSTF) problems, where models shift the attention on valuable data by increasing the weight.  

There are notable models focsing on the less explored and challenging long-term time series forecasting(LTSF) problem, include Log- Trans, Informer, Autoformer, Pyraformer, Triformer and the recent FEDformer. <d-cite key="Zeng_Chen_Zhang_Xu_2023"></d-cite> The Informer model applies ProbSparse self-attention mechanism to let each key to only attend to several dominant queries and then use the distilling operation to deal with the redundance. The operation privileges the superior ones with dominaitng features and make a focused self-attention feature map in the next layer, which trims the input's time dimension.<d-cite key="Zhou_Zhang_Peng_Zhang_Li_Xiong_Zhang_2021"></d-cite> Besides, additional position encoding can help the model to understand the periodicity inherented in traffic data, which implies applying the relative or global positioin encoding interms of weeks and days. <d-cite key="https://doi.org/10.1111/tgis.12644"></d-cite>


## Experiment


### Dataset
We used a multivariate traffic( https://pems.dot.ca.gov/) dataset records the road occupancy rates from different sensors on San Francisco freeways. We selected first 100 censors as our experiment dataset. 

## Result

## Discussion and Analysis


In addition to transformer architecture, a combination of various methods or framework may help us to benefit from the advantages of different models. The transformer-based framwork for multivariate time series representation lerning is proposed by George et al.  <d-cite key="DBLP:journals/corr/abs-2010-02803"></d-cite> The Spatial-Temporal Graph Neural Networks(STGNNs) is another widely used model in traffic prediction, which only consider short-term data. The STEP model is propsde to enhance STGNN with a scalable time series pre-training mode. In the pre-training stage. They split very long-term time series into segments and feed them into TSFormer, which is trained via the masked autoencoding strategy. And then in the forecasting stage. They enhance the downstream STGNN based on the segment-level representations of the pre-trained TSFormer.<d-cite key="10.1145/3534678.3539396"></d-cite>


<!-- ## Citations

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

*** -->

<!-- ## Footnotes

Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote> -->

<!-- ***

## Code Blocks

This theme implements a built-in Jekyll feature, the use of Rouge, for syntax highlighting.
It supports more than 100 languages.
This example is in C++.
All you have to do is wrap your code in a liquid tag:

{% raw  %}
{% highlight c++ linenos %}  <br/> code code code <br/> {% endhighlight %}
{% endraw %}

The keyword `linenos` triggers display of line numbers. You can try toggling it on or off yourself below:

{% highlight c++ %}

int main(int argc, char const \*argv[])
{
string myString;

    cout << "input a string: ";
    getline(cin, myString);
    int length = myString.length();

    char charArray = new char * [length];

    charArray = myString;
    for(int i = 0; i < length; ++i){
        cout << charArray[i] << " ";
    }

    return 0;
}

{% endhighlight %}


<!-- ## Blockquotes

<blockquote>
    We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another, unevenly. We grow partially. We are relative. We are mature in one realm, childish in another.
    —Anais Nin
</blockquote>

*** -->


<!-- ## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body` sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div> -->

<!-- ***

## Other Typography?

Emphasis, aka italics, with *asterisks* (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
⋅⋅* Unordered sub-list. 
1. Actual numbers don't matter, just that it's a number
⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)

* Unordered list can use asterisks
- Or minuses
+ Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[I'm a relative reference to a repository file](../blob/master/LICENSE)

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links. 
http://www.example.com or <http://www.example.com> and sometimes 
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style: 
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```
 
```python
s = "Python syntax highlighting"
print s
```
 
```
No language indicated, so no syntax highlighting. 
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the 
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote. 


Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a *separate paragraph*.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the *same paragraph*. -->
