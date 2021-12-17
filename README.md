# network_science_project

This repo contains the code for the final project of the network science class at UZH Fall 2021.

## Team Members: 
Laurin van den Bergh (16-744-401), Zheng Luo (21-738-901), Marco Di Panfilo (21-710-850), Lorenzo Spoleti (15-980-477)


## Abstract of the project:

The Coronavirus (COVID-19) outbreak has caused a global economic and financial crisis, especially when the
first wave hit in February 2020. The stock market suffered a crash after a decade-long rally. This study aims to
verify if COVID-19 affects the crypto market in a similar manner as the stock market. Thus the study focuses on
the 57 largest cryptocurrencies from January 1st, 2020 to November 26th, 2021. To avoid model bias brought
by using a single network for the analysis, we employ three different types of networks, Minimum Spanning
Tree, Threshold Network, and Planar Maximally Filtered Graph to examine the effect of COVID-19 on the crypto
market and the change of market structure over time. The findings reveal that the topology of the crypto market
changed towards a more compact and densely connected structure during the initial weeks of the pandemic with
an increased clustering coefficient, a shrunken average distance and a lower mean occupation layer for the MST,
which turns out to be the most informative network.


## Contribution and structure of repository:

Contribution by topic (topics sorted in chronological order) are given in the table below. All team members
agreed that the workload was distributed fairly and the conclusions were discussed and reached as a group
during regular group meetings

| Topic                              | Contributor          | File                              |
|------------------------------------|----------------------|-----------------------------------|
| Proposal                           | Laurin van den Bergh |                                   |
|                                    | Zheng Luo            |                                   |
| Coordination                       | Laurin van den Bergh |                                   |
| Exploratory Data Analysis          | Laurin van den Bergh | 1-EDA.ipynb                       |
|                                    | Zheng Luo            | 1.1-Data cleansing EDA.ipynb      |
| Data Preparation                   | Laurin van den Bergh | 1-EDA.ipynb, corrmat functions.py |
| Maximum Likelihood Estimation of Q | Laurin van den Bergh | 2-mle of Q.ipynb                  |
| Denoising Correlation Matrices     | Lorenzo Spoleti      | 3-denoise and mst.ipynb           |
| Mininum Spanning Tree              | Lorenzo Spoleti      | 3-denoise and mst.ipynb           |
| Threshold Network                  | Zheng Luo            | 3.1-threshold networks.ipynb      |
| Planar Maximally Filtered Graph    | Marco Di Panfilo     | 3.2-PMFG networks.ipynb           |
| Network Property Analysis          | Laurin van den Bergh | 4-network properties.ipy          |
|                                    | Lorenzo Spoleti      |                                   |

