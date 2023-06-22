# Bachelor Thesis Code
Welcome to the repository for the code developed as part of my Bachelor's thesis. This code accompanies my thesis titled "Lookback Option Pricing with the COS Method".

## Overview
In this repo you will find the following folders:
+ density_approximation: an early demonstration of the COS Method Density approximation procedure
+ montecarlo: a demonstration of monte carlo density approximation for lookback options
+ option_pricing: the implementation of the cos method for european and lookback option pricing as in chapter 3 of my thesis
+ option_pricing_extended: the implementation of cos method lookback option pricing via the spitzer recursion algorithm, as in chapter 4 of my thesis

## Installation
To run the code locally, please follow these steps:
1. Clone this repository to your local machine using
```
git clone https://github.com/JordPBvE/bachelor-project.git
```
2. Make sure you have python installed, if not, see https://wiki.python.org/moin/BeginnersGuide/Download
3. Install the required dependancies, either globally or in a virtual environment (for virtual environment, see https://docs.python.org/3/library/venv.html)
```
pip install numpy matplotlib scipy colorama winsound
```

## Usage
To run one of the four programs, navigate into the correct folder in your terminal and run
```
python main.py
```

## Results
Results to simulations can be found in the program folders. Gemerally, the most impoertant results are saved into `fig.png` images, and additional results, often used in testing, are inserted under names explaining the content.

