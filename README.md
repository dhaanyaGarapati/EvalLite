# EvalLite HSR – Human Subjects Research Study on LLM Evaluation

EvalLite HSR is a web-based evaluation tool built to support an IRB-compliant Human Subjects Research (HSR) study on how people perceive and compare responses from two modern Large Language Models (LLMs): GPT-4o and Claude 3 Haiku.

This application is designed to systematically collect:
- Participant-generated prompts  
- Paired model outputs (A and B)  
- Automated quality metrics (fluency, factuality, judge scores)
- Human ratings collected separately in a Qualtrics survey  

The purpose is to understand "how well automated LLM metrics align with real human judgments" of response quality.

## Purpose of the Study

This study investigates three major research goals:

### 1. Compare two LLMs (GPT-4o vs Claude Haiku)  
Participants provide 8 prompts across four domains. For each prompt, the app generates two responses (Output A and Output B). The model order (A or B) is randomly assigned to avoid bias.

### 2. Collect human ratings on fluency, factuality, and overall preference 
Participants rate each pair in a Qualtrics survey using standardized 7-point Likert scales to measure:
- Fluency  
- Factuality  
- Overall preference  

These ratings serve as the ground truth human evaluation.

### 3. Compare human scores with automated model metrics 
The system automatically computes:
- Rule-based fluency metrics  
- Rule-based factuality metrics  
- (Optional) Llama-based fact checking  
- (Optional) neutral judge assessments  

This enables testing whether automated LLM evaluation metrics correlate with or predict human ratings (alignment analysis).

The ultimate research goal is to determine whether automated evaluation can reliably reflect human judgment.

## How the Study Works (Participant Workflow)

### Step 1 — Enter email  
Used only for verification and linking to Qualtrics.

### Step 2 — Receive anonymized UID 
A hashed ID is automatically generated. No personal information is stored.

### Step 3 — Provide 8 prompts  
Participants enter two prompts for each domain:
- Biology  
- Technology  
- Science  
- Geography  

Each domain collects Prompt 1 and Prompt 2, for a total of 8 prompts.

### Step 4 — See paired outputs (A and B)  
For every prompt:
- The app generates two responses  
- It never reveals which model produced which output  
- Automated scores are computed internally but not shown  

The participant simply reviews the content.

### Step 5 — After all 8 prompts → Redirect to Qualtrics 
The app passes:
- UID  
- Prompt text (p1–p8)  
- Output A (A1–A8)  
- Output B (B1–B8)  

Qualtrics pulls these values into the survey using a Web Service request.

### Step 6 — Participant rates responses in Qualtrics
For each of the 8 items, participants rate:
- Fluency (1–7 Likert)  
- Factuality (1–7 Likert)  
- Overall preference (A / B / Tie)  

These serve as the human evaluation dataset.

## What Data This App Collects

The app does **NOT** collect personal information.  
It only stores:

### Participant Inputs
- 8 user prompts  
- Anonymous UID  
- Email (for linking only, not analyzed)

### Model Outputs
- Output A for each prompt  
- Output B for each prompt  
- Model identity hidden from participant

### Automated Metrics (for research only)
- Fluency score  
- Factuality score  
- Optional Llama fact-check  
- Optional judge scores

All data is stored locally.

## Why This Matters (Academic Motivation)

This project is part of a controlled HSR study in CS 690 — Evaluating Generative AI Systems.  
It addresses several key research questions:

- Do humans prefer outputs from one model over another?
- Do rule-based evaluation metrics match human perception?
- Can automated scoring be used to approximate or replace human judgments?
- Which LLM produces more fluent and factual results across different domains?

The study’s structure allows:
- Causal reasoning (model -> output quality -> human rating)
- Correlational analysis (automated metrics <--> human ratings)
- Generalization across domains

## Technologies Used

- Streamlit — Front-end interface  
- OpenAI GPT-4o — Model A/B candidate  
- Anthropic Claude 3 Haiku — Model A/B candidate  
- Python — Evaluation logic and metric computation  
- Qualtrics — Human ratings collection  
- CSV Logging — For later analysis  

## Study Outputs

This system produces several datasets suitable for academic analysis:
- Human ratings from Qualtrics
- Model outputs from Streamlit
- Automated model metrics
- Order randomization logs
- Participant prompts (qualitative data)


## Acknowledgments

Developed as part of CS 690: Evaluating Generative AI Systems
Instructor: Prof. David Porfirio  
George Mason University
