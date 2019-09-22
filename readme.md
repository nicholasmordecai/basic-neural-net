# Predict Gender From Height & Weight with Neural Network
NodeJS (TypeScript) implementation of some very basic machine learning with 2 inputs, 2 hiden neurons and one output neuron. 

## Get Running

This project has no dependencies, mathematical or machine learning. The only dependencies are for development:
-- **Dependencies**
* TypeScript
* Nodemon

-- **Types**
* @types/node

> this will keep alive until any .ts file is changed, the re-compile and nodemon will restart the process.
```sh
$ npm install
$ npm run dev
```

## How It Works
There are 2 inputs, two neurons in the hidden layer and 1 output neuron. Looks like this:

![alt text](https://victorzhou.com/network3-27cf280166d7159c0465a58c68f99b39.svg "Victor Zhou - Neural Network View")

Starting our neurons, weights and biases with random normal distributions, we apply our training data for the network to refine it's ability to accuratly predict the outcome.

We take the weight and height of a person, and with the trained network, we can accuratly predict the gender of any person. Of course this isn't always true as there are many people who have the same weight and height as the other gender. This is simply based on statistics.

## About
Just a fun little project I did to get an introduction to machine learning & neural networks. 

> Project is based from a Python implementation: -  [intro-to-neural-networks](https://victorzhou.com/blog/intro-to-neural-networks/).