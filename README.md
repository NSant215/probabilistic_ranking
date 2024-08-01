# probabilistic_ranking
Code to explore learnings in probabilistic ranking methods using match data of top tennis players.

This project outlines code to solidify learnings on how different rankings methods work, by conducting investigations on a dataset of 1801 matches played between the top 107 male tennis players in 2011.

Helper functions are available in python files, with the main code in the Jupyter notebook showing each step of the investigation. Results are shown and explained in the Report.

## Task One: Using Gibbs Sampling to estimate the joint distribution of player skills.

Gibbs Sampling is a probabilistic sampling technique used to estimate the joint distribution of multiple variables. It is particularly useful when the joint distribution is complex and difficult to sample directly. The goal of this investigation is to predict the outcome of matches between tennis players by sampling from a developing skill matrix.

### Gibbs Sampling steps:
1. Initialize the variables: In Gibbs Sampling, we start by initializing the variables we want to estimate. In this case, this is the skill of each tennis player.
2. Iterate through the variables: In each iteration, we select one variable and update its value based on the current values of the other variables. This is done by sampling from the conditional distribution of the selected variable given the values of the other variables.
3. Repeat the iterations: We repeat the iterations for a certain number of times or until convergence is reached. Convergence means that the estimated values of the variables stabilize and do not change significantly with further iterations. We measure convergence by when the autocorrelations of the samples are low (since Gibbs Sampling is a Markov Chain Monte Carlo method, this requires thinning) and stationary.
4. Collect samples: As we iterate through the variables, we collect samples of their values. These samples represent possible configurations of the variables that are consistent with the joint distribution.
5. Estimate the joint distribution: Finally, we can use the collected samples to estimate the joint distribution of the variables. This can be done by analyzing the frequency of different configurations in the samples.

## Task Two: Use Expectation Propogation to estimate the joint distribution of player skills by a marginal gaussian distribution for each player

Expectation Propagation is a method for approximating the marginal distribution of a set of variables in a probabilistic model. It is particularly useful when the joint distribution is complex and difficult to sample directly. The goal of this investigation is to predict the outcome of matches between tennis players by approximating the joint distribution of player skills using a marginal Gaussian distribution for each player.

### Expectation Propagation steps:
1. Initialize the variables: In Expectation Propagation, we start by initializing the variables we want to estimate. In this case, the marginal mean and precision of the skill of each tennis player.
2. Iterate through the variables: In each iteration, we use message passing of the marginal Gaussian distributions to update the marginal mean and precision of each player's skill.
3. Repeat the iterations: We repeat the iterations for a certain number of times or until convergence is reached. Convergence means that the estimated values of the variables stabilize and do not change significantly with further iterations.

## Task Three: 