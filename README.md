Repository Overview: This repository documents my capstone project for the Professional Certificate in Machine Learning and Artificial Intelligence. The project focuses on applying machine learning techniques to a Black-Box Optimisation (BBO) challenge, where the underlying functions and evaluation criteria are hidden.

**Black-Box Optimisation Project**

<ins>1. Project Overview</ins>

Imagine trying to find the highest point in a landscape, except you cannot see the landscape. You can choose a location and are told its height, but then you must use that information to decide where to look next. With only a limited number of attempts, every choice matters.

This is essentially the challenge behind this Black-Box Optimisation (BBO) capstone.

I am given eight unknown functions, ranging from 2 to 8 dimensions. I cannot see their formulas or gradients and can only learn about them through previous input-output observations and the results of my own queries.

The goal is to find inputs that produce the highest possible outputs using a limited number of evaluations. This reflects real-world problems where experiments can be expensive or time-consuming, such as tuning ML models or optimising chemical processes.

For my future career, this project is helping me develop both my machine learning knowledge and my ability to make evidence-based decisions with incomplete information.

<ins> 2. Inputs and Outputs </ins>

Each function receives numerical inputs between 0 and 1:

F1–F2: 2 inputs
F3: 3 inputs
F4–F5: 4 inputs
F6: 5 inputs
F7: 6 inputs
F8: 8 inputs

Queries are submitted as hyphen-separated values to six decimal places, for example:

`0.241663-0.206735-0.396859`

Each query returns one numerical output representing performance. Higher is always better, although some functions return negative values, where values closer to zero represent better performance.

<ins> 3. Challenge Objectives </ins>

The objective is to maximise all eight functions while making each query as useful as possible.

The difficulty is that I can only submit one new query per function each week, the underlying functions are unknown, and the available datasets are relatively small. The functions may also behave differently: some may be smooth, while others may be noisy, sensitive or contain several local maxima.

This creates the main question behind my strategy: Do I exploit an area that I already know performs well, or explore an uncertain area that could perform even better?

<ins> 4. Technical Approach </ins>

My main approach uses Gaussian Process (GP) regression as a surrogate for each unknown function. The GP estimates both the expected performance of a new input and the uncertainty surrounding that prediction.

I compare three acquisition functions: Probability of Improvement (PI) for more cautious exploitation, Expected Improvement (EI) for balancing improvement and uncertainty, and Upper Confidence Bound (UCB) for greater exploration.

I also compare RBF, Matérn and Rational Quadratic kernels with different length scales using leave-one-out cross-validation, allowing my model choices to evolve as I collect more data.

How my strategy has evolved

**Round 1** – Trusting the model:
I initially relied heavily on GP predictions and acquisition functions to select my queries.

**Round 2** – Learning from results:
I began comparing predictions against actual outputs and using these results to identify promising regions. This produced several new bests. For example, F5 improved from approximately 1,089 → 1,192 → 1,720.

**Round 3** – Combining the model with my own judgement:
I now treat GP predictions as evidence rather than instructions. I am starting to trust patterns in my observations alongside the model, particularly because this produced better results in the previous round.

I therefore don't apply the same strategy to every function. Where I have repeatedly found strong results, I focus more on local exploitation. For less understood functions, I deliberately explore or design queries to test patterns I have noticed.

Ultimately, I treat every query as both an opportunity to improve my score and an experiment that teaches me something new about the hidden function.

<ins> 5. Updated strategy development </ins>

My optimisation strategy has evolved throughout the BBO challenge as more observations have become available. Rather than applying the same method to all eight functions, I use a function-specific approach based on previous results, model validation and the behaviour observed in each search space.

Gaussian Processes remain my main surrogate model, but I treat their predictions and uncertainty as evidence rather than instructions. Depending on the function, I have combined GP modelling with acquisition functions such as UCB, EI and PI, sensitivity analysis, controlled local refinement and manual exploration.

I have also experimented with neural network classification and regression using TensorFlow. These models are only considered for query selection when their validation performance provides evidence that they add value over simpler approaches.

A separate optimisation journal records how this strategy has developed across each round, including:
- submitted queries
- reasoning behind each query
- predicted outputs where applicable
- actual returned results
- prediction gain/loss
- evaluation of the strategy
- decisions for the following round

[View my full BBO optimisation journal](https://docs.google.com/spreadsheets/d/1eA75HP5KTTgQZAAiwdnpb1mCnqJXC4tdzsE3-yhwvhY/edit?usp=sharing)

<ins> 6. Repository structure </ins>

The `/stage-two` directory is organised so that each BBO round has its own analysis notebook. This preserves the analysis and decisions made at each stage of the optimisation process and makes it easier to follow how the strategy develops over time.

`query_history.py` acts as the central record of submitted queries and returned outputs. Each new round loads the initial observations together with this accumulated query history.

Reusable functionality has been moved into `bbo_utils.py`, including data loading, Gaussian Process fitting and validation, acquisition functions, result checking, visualisation and query formatting. This keeps the individual round notebooks focused on analysis and query decisions rather than repeating implementation code.

The main modelling and analysis libraries used are:

- NumPy and pandas for data handling
- Matplotlib for visualisation
- SciPy for acquisition calculations
- scikit-learn for Gaussian Processes, kernel evaluation and model validation
- TensorFlow for experimental neural network models