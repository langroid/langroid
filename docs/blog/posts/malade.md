---
title: 'MALADE: Orchestration of LLM-powered Agents with Retrieval Augmented Generation for Pharmacovigilance'
draft: true
date: 2024-07-16
authors:
- jihye
- nils
- pchalasani
- mengelhard
- someshjha
- anivaryakumar
- davidpage

categories:
- langroid
- multi-agent
- neo4j
- rag
comments: true
---

# MALADE: Orchestration of LLM-powered Agents with Retrieval Augmented Generation for Pharmacovigilance

[Arxiv](tbd) 

[GitHub](https://github.com/jihyechoi77/malade)

## Summary
We introduce MALADE (**M**ultiple **A**gents powered by **L**LMs for **ADE** Extraction),
a multi-agent system for Pharmacovigilance. It is the first effective explainable 
multi-agent LLM system for extracting Adverse Drug Events (ADEs) from FDA drug labels and drug prescription data.
<!-- more -->
Given a drug category and an adverse outcome, MALADE
produces a qualitative label ("increase/decrease risk" or "no effect"), a justification with citations,
as well as quantitative measures such as Confidence in the label, Probability,
Frequency, and Strength of Evidence.
This task is challenging for several reasons: (a) FDA labels and prescriptions are for individual drugs,
not drug categories, so representative drugs in a category need to be identified from patient prescription 
data, and ADE information found for specific drugs in a category needs to be aggregated to 
make a statement about the category as a whole, (b) the data is noisy, with variations in
the terminologies of drugs and outcomes, and (c) ADE descriptions are often buried
in large amounts of narrative text. 
The MALADE architecture is LLM-agnostic 
and leverages the [Langroid](https://github.com/langroid/langroid) multi-agent framework.
It consists of a combination of Agents using Retrieval Augmented Generation (RAG) and Critic Agents
that provide feedback to the RAG agents. 
We evaluate the quantitative scores against 
a ground-truth dataset known as the *OMOP Ground Truth Task* and find that MALADE achieves state-of-the-art 
performance.



## Introduction

Pharmacovigilance is a critical task in healthcare, where the goal is to monitor and evaluate the safety of drugs.
In particular, the identification of Adverse Drug Events (ADEs) is crucial for ensuring patient safety. Consider a 
question such as this:

> What is the effect of **ACE inhibitors** on the risk of developing **angioedema**?

Here the **drug category** $C$ is _ACE inhibitors_, and the **outcome** $O$ is _angioedema_.
Answering this question involves these steps:

1. (a) **Find all drugs** in the ACE inhibitor category $C$, e.g. by searching the FDA 
[National Drug Code](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory) (NDC) 
   database. This can be done by a SQL query, with additional filters to handle variations in drug/category names 
   and inaccurate classifications.
1. (b) **Find the prescription frequency** of each drug in $C$ from patient prescription data, e.g. the [MIMIC-IV]
   (https://physionet.org/content/mimiciv/2.2/) database. This can again be done by a SQL query.
1. (c) **Identify the representative drugs** $D \subset C$ in this category, based on prescription frequency data from 
   step 2.  
2. For each drug $d \in D$, **summarize ADE information** about the effect of $d$ on the outcome $O$ of interest,
   (in this case angioedema) from text-based pharmaceutical sources, 
    e.g. the [OpenFDA Drug Label](https://open.fda.gov/apis/drug/label/) database.
3. **Aggregate** the information from all drugs in $D$ to make a statement about the category $C$ as a whole.


## The role of LLMs

While steps 1(a) and 1(b) can be done by straightforward SQL queries, the remaining steps are challenging but ideally 
suited to LLMs:

### Step 1(c): Identifying representative drugs in a category from prescription frequency data (`DrugFinder` Agent)

This is complicated by noise, such as the same drug appearing multiple times under different names, 
formulations or delivery methods (For example, the ACE inhibitor **Lisinopril** is also known as **Zestril** and **Prinivil**.) 
  Thus a judgment must
  be made as to whether these are sufficiently different to be considered pharmacologically distinct;
  and some of these drugs may not actually belong to the category. This task thus requires a grouping operation, 
  related to the task of identifying standardized drug codes from text descriptions,
  well known to be challenging. This makes it very difficult to explicitly define in a deterministic 
  manner that covers all edge cases (unlike the above database tasks), and hence is well-suited
  to LLMs, particularly those such as GPT-4, Claude3.5, and similar-strength variants which are known to have been 
  trained on vast amounts of general medical texts. 

In MALADE, this task is handled by the `DrugFinder` agent,
which is an Agent/Critic system where the Critic agent helps improve the main
agent’s output via iterative feedback; in particular, the Critic corrects the Agent when it incorrectly
classifies drugs as pharmacologically distinct

###  Step 2: Identifying Drug-Outcome Associations (`DrugOutcomeInfoAgent`)

The task here is to identify whether a given drug
has an established effect on the risk of a given outcome, based on FDA drug label database, and
output a summary of relevant information, including the level of identified risk and the evidence for
such an effect. Since this task involves extracting information from narrative text, it is well-suited to
LLMs using the Retrieval Augmented Generation (RAG) technique. 

In MALADE, the `DrugOutcomeInfoAgent` handles this task, and is also an Agent/Critic system, where the Critic
provides feedback and corrections to the Agent's output.
This agent does not have direct access to the FDA Drug Label data, but can receive
this information via another agent, `FDAHandler`. FDAHandler is equipped with **tools** (also known as function-calls) 
to invoke the OpenFDA API for drug label data, and answers questions in the context of information retrieved
based on the queries. Information received from this API is ingested into a vector database, so the
agent first uses a tool to query this vector database, and only resorts to the OpenFDA API tool if
the vector database does not contain the relevant information. An important aspect of this agent is that
its responses include specific **citations** and **excerpts** justifying its conclusions.

###  Step 3: Labeling Drug Category-Outcome Associations (`CategoryOutcomeRiskAgent`)

To identify association between a drug category C and an adverse health outcome $O$, we concurrently run a batch of 
queries to copies of `DrugOutcomeInfoAgent`, one for each drug $d$ in the
representative-list $D$ for the category, of the form: 

> Does drug $d$ increase or decrease the risk of condition $O$?

The results are sent to `CategoryOutcomeRiskAgent`, 
which is an Agent/Critic system which performs the final classification
step; its goal is to generate a label identifying whether a category of drugs

- **increases** the risk, 
- **decreases** the risk, or 
- has **no effect** on the risk of the outcome $O$.

In addition to this qualitative label, `CategoryOutcomeRiskAgent`
produces a number of additional outputs:

- **Confidence** in the generated label, as a number in $[0,1]$.
- **Probability** (in $[0,1]$) that the category $C$ has _any_ effect (positive or negative) on the risk of outcome $O$.
- **Strength of Evidence** for the label, as  _none_, _weak_, or _strong_.
- **Frequency of effect**, as _rare_, _common_, or _none_.

## MALADE Architecture

The figure below illustrates how the MALADE architecture handles the query,

> What is the effect of **ACE inhibitors** on the risk of developing **angioedema**?

![malade-arch.png](figures/malade-arch.png)

The query triggers a sequence of subtasks performed by the three Agents described above: 
`DrugFinder`, `DrugOutcomeInfoAgent`, and `CategoryOutcomeRiskAgent`.
Each Agent generates a response and justification, which are validated by a corresponding Critic agent, whose feedback is
used by the Agent to revise its response.

## Evaluation

We evaluate the results of MALADE against a well-established ground-truth dataset, 
the [OMOP ADE ground-truth table](https://www.niss.org/sites/default/files/Session3-DaveMadigan_PatrickRyanTalk_mar2015.pdf), shown below.
This is a reference dataset within the Observational Medical Outcomes Partnership (OMOP) Common Data Model that 
contains validated information about known adverse drug events.


![omop-ground-truth.png](figures/omop-ground-truth.png)

Below is a side-by-side comparison of this ground-truth dataset (left) with MALADE's labels (right), ignoring blue 
cells (see the paper for details):

![omop-results.png](figures/omop-results.png)

The resulting confusion-matrix for MALADE is shown below:

![confusion.png](figures/confusion.png)

To evaluate the quantitative scores produced by MALADE, we use the AUC metric.
We also perform ablations to identify whether (and by how much) the various components of the
MALADE system contribute to its performance. Please see the paper for details.




