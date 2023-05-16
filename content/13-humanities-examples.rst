.. _example-humanities:

Use cases: humanities
==============================

.. questions::

   - How are GPUs used in practical humanities problems?
   - Why are they necessary in fields outside the hard sciences?

.. objectives::

   - o1
   - o2

.. instructor-note::

   - X min teaching
   - X min exercises


.. keypoints::

GPU computing in the humanities

When most people think about GPU computing, they often consider it a completely separate field of study - which it is. This can put most people off of looking into it as a solution to their problems because they would have to put aside a few weeks - or months - to learn it. Such a strategy would not be practical for the average researcher or anyone considering it. The truth is that most of the work being done in the humanities is based on using deep learning models which have access to GPUs for training and inference. You do not need to be an expert to use pretrained models or use GPUs for inference with those models. Instead, you need to have a rough idea of how the system works, so that in those cases, you can reach out to someone who has the appropriate expertise who can help you with specific problems.

Since a lot of the current academic and professional landscape is becoming interdependent e.g. humanities research is relying a lot on the developments in computer science. We need to consider the fact it isn't necessarily the case that each department needs to become experts all involved fields, but rather that you need to have an understanding of how those fields impact yours, so that you can reach out to the appropriate people. For example, if I were a researcher in the humanities who wants to use a pretrained deep learning model for analyzing image data, then I don't need to be an expert in GPU computing. I simply need to be able to run my model's inference on a GPU, so the specific problem that I'm trying to solve is porting my current code to the GPU.

Language models and NLP

With the recent popularity of ChatGPT, the use of language models has come into the mainstream, however such models have been used in the humanities many years already. One of the biggest goals of humanities researchers is working with textual data which has increased exponentially over recent years due to the rise in social media. Analyzing such textual data to gain insights into questions of sociology, linguistics and various other fields have become increasingly reliant on using language models. Along with language models, the need for GPU access has become essential.

Archeology

Pivoting to a slightly different use case, the field of archeology also makes use of GPUs in their 3d modelling and rendering work. The biggest problem with archeological sites is that once they are excavated, they are destroyed, so any researchers who aren't present at the site, would lose valuable insights into how it looked when it was found. However, with recent developments in technology and accessibility to high-performance computing, they are able to generate extremely detailed renderings of the excavation sites which act as a way to preserver the site for future researchers to gain critical insights and contribute to the research. One such institution is the DarkLab at Lund University which has made available their digital collections on their website for anyone to see.

Access to GPUs for inference

As was mentioned briefly in the introduction, most researchers do not need to be trained in the field of GPU computing, but rather only require access to GPUs when doing inference. For example, when analyzing large numbers of texts, the speed at which you can do that when you have access to GPUs makes it much more reasonable. These types of problems are not usually very labour intensive, but rather the biggest issue is the expertise to know that using a GPU is an available option. In that case, these problems are low hanging fruit which can be solved relatively easily.
