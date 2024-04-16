---
title: Claim Verification in NLP
---

<b>Teachers:</b> Tatiana Anikina and Natalia Skachkova (DFKI)

<b>Description:</b>

This seminar is about <b>claim verification</b>, which is an increasingly important topic in NLP. Claim verification helps to combat misinformation and for this we need to (1) identify claims that can be verified, (2) retrieve relevant evidence and (3) make a prediction (e.g., fake vs true) with (optional) justification.

This task can be addressed in different ways and we will start by providing a brief overview of the claim verification research area, introduce some terminology, commonly used approaches and datasets. Then we will delve into more details and students will present some recent papers on this topic (see the recommended selection of papers below).

<b>Organizational info:</b>

This is a <b>seminar</b> with 2-3 introductory lectures at the beginning of the semester followed by the student presentations.
 
Each participant will be asked to give a 20-25 minutes presentation on one of the papers and prepare some questions for the papers presented by others. Active participation in the discussion is encouraged and counts towards a better grade. The seminar also offers an opportunity to earn 7 CP (credit points) if the student submits a term paper after the presentation (only presentation gives 4 CP).

<b>Sessions:</b> 

The first sessions will be on Thursday (18.04) at 8:30 in C7.3 Seminar room 1.12. Please note that we may change the seminar slot depending on the preferences, we will discuss this in the first session.

<b>Location:</b> C7.3 Seminar room 1.12


<b>Teams channel:</b> [Link to the MS Teams channel]

[Link to the MS Teams channel]: https://teams.microsoft.com/l/team/19%3apMiFIBD39G4ffkzbRfpKJfFBAPnanJWYN-hgCtSq6dM1%40thread.tacv2/conversations?groupId=80931fce-c6dd-485f-b7cd-a125b3426ba7&tenantId=67610027-1ac3-49b6-8641-ccd83ce1b01f

<b>Requirements:</b>

4 CP: 20 min. presentation, active participation in the paper discussions

7 CP: same as for 4 CP plus a term paper about the topic presented in the seminar

<b>Office hours:</b>

We will also offer office hours during the semester. Please write us an email to make an appointment (first_name.last_name@dfki.de). Students are welcome to ask any questions concerning their presentation and the papers from the reading list.

### <b>Recommended reading list:</b>

#### <b>Datasets and Annotation</b>

1. HoVer: A Dataset for Many-Hop Fact Extraction And Claim Verification

   Link: <https://aclanthology.org/2020.findings-emnlp.309>

   <details>
   <summary>Summary</summary>
   <ul>
   <li>A <b>multi-hop</b> dataset whose claims require evidence from as many as four English Wikipedia articles</li>
   
   <li><b>A pipeline system</b> of fact extraction and claim verification
    <ul>
    <li>Rule- and neural-based document retrieval</li>
    <li>Neural-based sentence selection</li>
    <li>BERT-based model for recognizing entailment between a claim and an set of sentences (called evidence)</li>
    </ul>
   </li>
   </ul>
   </details>
<br />

2. DialFact: A Benchmark for Fact-Checking in Dialogue
   
   Link: <https://aclanthology.org/2022.acl-long.263>

   <details>
   <summary>Summary</summary> 
   A <b>dialogue dataset</b> with <b>three sub-tasks</b>:
   <ul>
    <li>Verifiable claim detection aims to distinguish responses that do not contain verifiable factual information</li>
    <li>Evidence retrieval involves selecting the most relevant knowledge snippets from Wikipedia which can verify the response</li>
    <li>Claim verification aims to classify if a response is supported, refuted, or does not have enough information to be verified given the dialogue history and the retrieved evidence</li>
   </ul>
  Lexical overlap-based <b>models</b> for the 1st sub-task; 2nd sub-task consists of document retrieval and sentence selection; in the 3rd sub-task claim is classified based on the context and selected sentences.
   </details>
<br />

3. AVERITEC: A Dataset for Real-world Claim Verification with Evidence from the Web
   
   Link: <https://proceedings.neurips.cc/paper_files/paper/2023/file/cd86a30526cd1aff61d6f89f107634e4-Paper-Datasets_and_Benchmarks.pdf>

   <details>
   <summary>Summary</summary>
   
   A new <b>dataset</b> of 4,568 real-world claims. Each claim is annotated with <b>question-answer pairs</b> supported by evidence available online, as well as <b>textual justifications</b> explaining how the evidence combines to produce a verdict.
   
   Baseline <b>pipeline model</b> (uses <b>prompting</b>):
   <ul>
   <li>Retrieve documents using claim's keywords and generated quetions</li>
   <li>Pick out sentences and generate question for them</li>
   <li>Produce labels</li>
   <li>Generate a textual justification for the verdict</li>
   </ul>
   </details>
<br />

4. FACTIFY: A Multi-Modal Fact Verification Dataset
   
   Link: <https://ceur-ws.org/Vol-3199/paper18.pdf>

   <details>
   <summary>Summary</summary>
   The largest <b>multimodal</b> fact verification public dataset consisting of 50K data points, covering news from India and the US. It has <b>five categories</b>: Support_Text, Support_Multimodal, Insufficient_Text, Insufficient_Multimodal and Refute. This paper explores 2 different settings to establish the baselines i.e., text-only & multimodal.
   </details>
<br />

5. EX-FEVER: A Dataset for Multi-hop Explainable Fact Verification
   
   Link: <https://arxiv.org/pdf/2310.09754.pdf>

   <details>
   <summary>Summary</summary>   
   A dataset for <b>multi-hop explainable</b> fact verification. Each claim is accompanied by two or three golden documents containing the necessary information for veracity reasoning, a veracity label and an explanation that outlines <b>the reasoning path supporting the veracity classification</b>.
   
   The <b>baseline model</b> includes 3 parts: 
   <ul>   
   <li>Rule- and neural-based document retrieval</li>
   <li>Document <b>summarization using BART</b></li>
   <li>Verdict prediction with BERT or a graph-based text reasoning model, the state-of-art fact-checking model GEAR</li>
   </ul>

   This paper explores <b>LLMs in the fact checking task</b> in two directions: directly using LLMs as an actor, and using LLMs as a planner, they also evaluate the verdict accuracy and the ability of LLMs to generate explanations.
   </details>
<br />

6. Active PETs: Active Data Annotation Prioritisation for Few-Shot Claim Verification with Pattern Exploiting Training
   
   Link: <https://aclanthology.org/2023.findings-eacl.14>

   <details>
   <summary>Summary</summary> 
   Focus is on <b>claim verification step</b>. This paper proposes to optimise the selection of candidate instances to be labelled through <b>active learning</b> in a situation when there is <b>a lack of annotated data</b>.
   </details>
<br />

#### <b>Approaches and Models</b>

7. Fact or Fiction: Verifying Scientific Claims
   
   Link: <https://aclanthology.org/2020.emnlp-main.609>

   <details>
   <summary>Summary</summary>   
   A <b>dataset</b> of 1,409 <b>scientific claims</b> accompanied by abstracts that support or refute each claim, and annotated with <b>rationales</b> justifying each SUPPORTS / REFUTES decision.
  
   The <b>baseline</b> is a pipeline system which:
   <ul>
   <li>Retrieves abstracts related to an input claim  using the TF-IDF score</li>
   <li>Uses a BERT-based sentence selector to identify rationale sentences by scoring them</li>
   <li>Labels each abstract as SUPPORTS, REFUTES, or NOINFO with respect to the claim, using BERT</li>
   </ul>
   </details>
<br />

8. Generating Fact Checking Explanations
   
   Link: <https://aclanthology.org/2020.acl-main.656>

   <details>
   <summary>Summary</summary>
   This is a study for how <b>justifications for verdicts</b> on claims can be <b>generated automatically</b> based on available claim context, and how this task can be modelled jointly with veracity prediction.  
   </details>
<br />

9. A Semantics-Aware Approach to Automated Claim Verification
    
   Link: <https://aclanthology.org/2022.fever-1.5>

   <details>
   <summary>Summary</summary>
   This work demonstrates that enriching a BERT model with <b>explicit semantic information</b> (Semantic Role Labelling, Open Information Extraction) helps to improve results in claim verification. Focus on <b>verdict prediction</b>. This approach integrates semantic information using the SemBERT architecture.
   </details>
<br />

10. Generating Literal and Implied Subquestions to Fact-check Complex Claims
    
    Link: <https://aclanthology.org/2022.emnlp-main.229>

    <details>
    <summary>Summary</summary>  
    Focus is on <b>decomposing a complex claim</b> into a comprehensive set of yes-no sub-questions whose answers influence the veracity of the claim.
    This paper presents a <b>dataset</b> of decompositions for over 1000 claims. Given a claim and its verification paragraph written by fact-checkers, they write subquestions covering both explicit propositions of the original claim and its implicit facets. Each claim is classified as one of <b>six labels</b>: pants on fire (most false), false, barely true, half-true, mostly true, and true.
    They also study whether SOTA pre-trained models can learn to <b>generate such subquestions</b> and do not build a full pipeline for fact verification in this paper.
    </details>
<br />

11. Explainable Claim Verification via Knowledge-Grounded Reasoning with Large Language Models
    Link: <https://aclanthology.org/2023.findings-emnlp.416>

    <details>
    <summary>Summary</summary>
    In this paper the authors attempt to verify <b>complex claims</b> and <b>generate explanations</b> without any annotated evidence, just by using <b>LLMs</b>. They leverage the in-context learning ability of LLMs to translate the claim into a <b>First-Order-Logic (FOL)</b> clause consisting of predicates, each corresponding to a subclaim that needs to be verified. Then, they perform FOL-Guided reasoning over a set of knowledge-grounded question-and-answer pairs to make veracity predictions and generate explanations to justify the decision-making process. The generated answers are grounded in real-world truth via retrieving accurate information from trustworthy <b>external knowledge sources</b> (e.g. Google or Wikipedia).
    </details>
<br />

12. Low-Shot Learning for Fictional Claim Verification
    
    Link: <https://www.semanticscholar.org/paper/Low-Shot-Learning-for-Fictional-Claim-Verification-Chadalapaka-Nguyen/8232d36767293ed7ba64c1feb6d1f80d6641fb9d>

    <details>
    <summary>Summary</summary>
     This work studies the problem of claim verification in the context of claims about <b>fictional stories</b> in a low-shot learning setting.
     The paper presents 2 <b>datasets</b>:
     <ul>
     <li>2000 fictional stories pulled from the r/WritingPrompts</li>
     <li>2000 r/stories from subreddits and sourced from Kaggle</li>
     </ul>
     
     Focus is on the detection of two main classes of plot holes: <b>continuity errors</b>, and <b>unresolved storylines</b>. The <b>pipeline</b> consists of 3 phases: two data preprocessing steps to first generate story encodings and then a knowledge graph, and then a joint graph neural network and deep neural network (DNN) model.
    </details>
<br />   

13. GERE: Generative Evidence Retrieval for Fact Verification
   
    Link: <https://dl.acm.org/doi/pdf/10.1145/3477495.3531827>

    <details>
    <summary>Summary</summary>
    The paper proposes to bypass the explicit retrieval process and introduces <b>a system that retrieves evidences in a generative way</b>. It exploits a transformer-based encoder–decoder architecture, pre-trained with a language modeling objective and fine-tuned to generate document titles and evidence sentence identifiers jointly.
    
     <ul>
     <li>Memory and computational cost is greatly reduced because the document index is eliminated and the heavy ranking process is replaced by a light generative process</li>
     <li>This approach considers the dependency information, which contributes to improving the consistency and eliminating duplication among the evidences</li>
     <li>Generative formulation allows to dynamically decide on the number of relevant documents and sentences for different claims</li>
     </ul>
    
    Based on the evidences obtained by GERE, they also train <b>a claim verification model</b>.
    </details>
<br />

14. Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-Step Prompting Method
    
    Link: <https://arxiv.org/pdf/2310.00305.pdf>

    <details>
    <summary>Summary</summary> 
     This work examines <b>LLMs</b> with in-context learning for news claim verification, and finds that only with 4-shot demonstration examples, the performance of several prompting methods becomes comparable with previous supervised models.
   
     The paper introduces <b>a prompting method</b> which directs LLMs to <b>separate a claim into several subclaims</b> and then <b>verify each of them via multiple questions-answering steps</b> progressively.
    </details>
<br />

### <b>Additional papers:</b>

#### <b>Datasets and Annotation</b>

1. SciFact-Open: Towards open-domain scientific claim verification
   Link: <https://aclanthology.org/2022.findings-emnlp.347>

2. WatClaimCheck: A new Dataset for Claim Entailment and Inference
   Link: <https://aclanthology.org/2022.acl-long.92.pdf>

#### <b>Approaches and Models</b>

3. Evidence-based Fact-Checking of Health-related Claims
    Link: <https://aclanthology.org/2021.findings-emnlp.297.pdf>

4. Claim-Dissector: An Interpretable Fact-Checking System with Joint Re-ranking and Veracity Prediction
    Link: <https://arxiv.org/pdf/2207.14116.pdf>

5. Missing Counter-Evidence Renders NLP Fact-Checking Unrealistic for Misinformation
    Link: <https://arxiv.org/pdf/2210.13865.pdf>

6. Synthetic Disinformation Attacks on Automated Fact Verification Systems
    Link: <https://ojs.aaai.org/index.php/AAAI/article/download/21302/21051>

7. Generating Scientific Claims for Zero-Shot Scientific Fact Checking
    Link: <https://arxiv.org/pdf/2203.12990.pdf%20>

8. Counterfactual Debiasing for Fact Verification
    Link: <https://aclanthology.org/2023.acl-long.374.pdf>

9. Fact-Checking Complex Claims with Program-Guided Reasoning
    Link: <https://arxiv.org/pdf/2305.12744.pdf>
