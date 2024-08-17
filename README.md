# Mitigating Gender Bias in Algorithmic Hiring with Deep Reinforcement Learning

Emilia Lautz

London School of Economics and Political Sciences 

Department of Methodology

August 15, 2024

## Abstract

In recent years, companies have increasingly deployed artificial intelligence (AI) within hiring, claiming
that automated hiring systems (AHS) save time, resources, and reduce human biases. However, AHS from
companies like Google and Amazon treated female and male applicants differently, raising concerns about
their impartiality. This paper investigates how gender fairness can be enhanced in algorithmic hiring by
developing a reinforcement learning based AHS. A double deep Q-network was trained to solve a hiring
problem, where an employer agent allocates jobs to candidates aiming at maximising the total value gained
by the firm. The gender distribution of the selected candidates was subsequently analysed. Results showed
that only 10% of hires were female, which was largely attributed to the gender imbalance in the training
data. Two strategies were implemented to address this: reducing the sample size disparities and adjusting the
reward function to encourage female employment. While raising the female representation in the training data
reduced their hiring rate, the specialised reward function successfully increased female hires. This research
demonstrates the importance of carefully defining fairness, as this understanding is inevitably adopted by
the AI, which risks perpetuating preexisting inequalities - but it also offers an opportunity to mitigate them.

## Dissertation PDF

## Repository Structure

```bash
    .
    ├── agents
    │   ├── agents_basic.py
    │   ├── agents_endorse.py
    ├── data 
        ├── analysis
        │   ├── cleaned_data.ipynb
        │   ├── raw_stackoverflow_data.ipynb
        ├── cleaned_data
        │   ├── candidates_cleaned.csv
        │   ├── jobs_cleaned.csv
        ├── raw_data
        │   ├── amazon_data_raw.csv
        │   ├── stackoverflow_data_raw.csv
    │   ├── amazon_data_cleaning.ipynb
    │   ├── amazon_data_scraping.py
    │   ├── stackoverflow_data_cleaning.ipynb
    ├── environment
    │   ├── environment.py
    ├── episodes
    ├── optimisation
    ├── README.md
    └── tools
```

## Code 

### Getting started


### Simulate the hiring game
