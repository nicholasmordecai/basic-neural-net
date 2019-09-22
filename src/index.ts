import { Neuron } from './Neuron';
import { Utils } from './utils';
import { readFileSync } from 'fs';

// Gender enum for easier selection between 0 and 1 respectivly
export enum Gender {
    'male', 
    'female'
}

export class NeuralNet {
    private trueData: number[];
    private trainingData: Array<{ weight: number, height: number }>;
    private neuron1: Neuron;
    private neuron2: Neuron;
    private outputNeuron: Neuron;
    private weightMean: number;
    private heightMean: number;

    constructor() {
        // Load the two data files we need
        this.trainingData = JSON.parse(readFileSync(__dirname + '/../data/training_data.json', 'utf8'));
        this.trueData = JSON.parse(readFileSync(__dirname + '/../data/true_results.json', 'utf8'));

        // The mean of the weight and height from the training data (pounds & inches)
        this.weightMean = 135;
        this.heightMean = 66

        // Create our three neurons here with random values initially
        this.neuron1 = new Neuron([Utils.randomNormalDist(), Utils.randomNormalDist()]);
        this.neuron1.bias = Utils.randomNormalDist();

        this.neuron2 = new Neuron([Utils.randomNormalDist(), Utils.randomNormalDist()]);
        this.neuron2.bias = Utils.randomNormalDist();

        this.outputNeuron = new Neuron([Utils.randomNormalDist(), Utils.randomNormalDist()]);
        this.outputNeuron.bias = Utils.randomNormalDist();

        // Begin training the network - 10k generations (epochs)
        this.trainNetwork(10000);
    }

    /**
     * @description Public function to call from outside of the class
     * 
     * @param weight 
     * @param height 
     */
    public predict(weight: number, height: number): number {
        const newInput: number[] = [weight - this.weightMean, height - this.heightMean];
        const result: number = this.feedForward(newInput);
        const gender: string = Gender[(result > 0.5 ? 1 : 0)];
        console.log(`Perdition for weight: ${weight} and height: ${height} is ${gender} with a certianty of ${Math.round((100 - (result * 100)))}%`);
        return result;
    }

    /**
     * @description Function that uses the networked neurons to predict an output
     * 
     * @param x 
     */
    private feedForward(x: number[]) {
        const result1: number = Utils.sigmoid(this.neuron1.weights[0] * x[0] + this.neuron1.weights[1] * x[1] + this.neuron1.bias);
        const result2: number = Utils.sigmoid(this.neuron2.weights[0] * x[0] + this.neuron2.weights[1] * x[1] + this.neuron2.bias);
        const output: number = Utils.sigmoid(this.outputNeuron.weights[0] * result1 + this.outputNeuron.weights[0] * result2 + this.outputNeuron.bias);
        return output;
    }

    /**
     * @description Loop through all training data for n epochs
     * 
     * @param epochs
     */
    private trainNetwork(epochs: number): void {
        // Define learning parameters
        const learnRate: number = 0.1;
        let epoch: number = 0;

        // Loop for n times where n = epochs
        for (let i = 0; epoch < epochs; i++) {
            // loop through each row if training data
            for (let i = 0; i < this.trainingData.length; i++) {

                // keep this just for shorthand later
                const x = this.trainingData[i];

                // calculate all the forward feeds from the neurons for later calculations
                const sumH1: number = this.neuron1.feedforward([x.weight, x.height]);
                const h1: number = Utils.sigmoid(sumH1);

                const sumH2: number = this.neuron2.feedforward([x.weight, x.height]);
                const h2: number = Utils.sigmoid(sumH2);

                const sumO1: number = this.outputNeuron.feedforward([h1, h2]);
                const o1: number = Utils.sigmoid(sumO1);
                const yPred = o1;

                const d_L_d_ypred: number = -2 * (this.trueData[i] - yPred);

                // Neuron o1
                const d_ypred_d_w5: number = h1 * Utils.derivativeSigmoid(sumO1);
                const d_ypred_d_w6: number = h2 * Utils.derivativeSigmoid(sumO1);
                const d_ypred_d_b3: number = Utils.derivativeSigmoid(sumO1);
                const d_ypred_d_h1: number = this.outputNeuron.weights[0] * Utils.derivativeSigmoid(sumO1);
                const d_ypred_d_h2: number = this.outputNeuron.weights[1] * Utils.derivativeSigmoid(sumO1);

                // Neuron h1
                const d_h1_d_w1 = this.trainingData[i].weight * Utils.derivativeSigmoid(sumH1);
                const d_h1_d_w2 = this.trainingData[i].height * Utils.derivativeSigmoid(sumH1);
                const d_h1_d_b1 = Utils.derivativeSigmoid(sumH1);

                // Neuron h2
                const d_h2_d_w3 = this.trainingData[i].weight * Utils.derivativeSigmoid(sumH2);
                const d_h2_d_w4 = this.trainingData[i].height * Utils.derivativeSigmoid(sumH2);
                const d_h2_d_b2 = Utils.derivativeSigmoid(sumH2);

                // Update neuron 1 weights & bias
                this.neuron1.weights[0] -= learnRate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1;
                this.neuron1.weights[1] -= learnRate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2;
                this.neuron1.bias -= learnRate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1;
                
                // Update neuron 2 weights & bias
                this.neuron2.weights[0] -= learnRate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3;
                this.neuron2.weights[0] -= learnRate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4;
                this.neuron2.bias -= learnRate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2;

                // Update output neuron weights & bias
                this.outputNeuron.weights[0] -= learnRate * d_L_d_ypred * d_ypred_d_w5;
                this.outputNeuron.weights[0] -= learnRate * d_L_d_ypred * d_ypred_d_w6;
                this.outputNeuron.bias -= learnRate * d_L_d_ypred * d_ypred_d_b3;
            }
            // only console out every 50 epochs
            if (epoch % 50 == 0) {
                const y_preds = this.applyAlongAxis();
                const loss: number = this.mseLoss(this.trueData, y_preds)
                console.log(`Epoch: ${epoch}, Loss: ${loss.toFixed(3)}`);
            }
            epoch++;
        }
    }

    /**
     * @description Run through an array of numbers and apply a function, pushing each result to a new array
     */
    private applyAlongAxis(): number[] {
        const result: number[] = [];
        for (let i = 0; i < this.trueData.length; i++) {
            result.push(this.feedForward([this.trainingData[i].weight, this.trainingData[i].height]));
        }
        return result;
    }

    /**
     * @description Mean Squared Error
     * 
     * @param trueValues
     * @param outputValues 
     */
    private mseLoss(trueValues: number[], outputValues: number[]): number {
        const diff = this.arrayDiff(trueValues, outputValues);
        return this.mean(diff);
    }

    /**
     * @description Calculate the difference (number) between two arrays looking at each matching index at a time
     * 
     * @param array1 
     * @param array2 
     */
    private arrayDiff(array1: number[], array2: number[]): number[] {
        const diff: number[] = [];
        for (let i = 0; i < array1.length; i++) {
            diff.push(Math.pow(array1[i] - array2[i], 2));
        }
        return diff;
    }

    /**
     * @description Calculate the mean of a single dimensional array
     * 
     * @param array
     */
    private mean(array: number[]): number {
        let total: number = 0;
        let count: number = array.length;
        for (const val of array) {
            total += val;
        }
        return total / count;
    }
}

// Construct our new neural net class
const neuralNet = new NeuralNet();

// Supply new inputs to see the expected output
const result: number = neuralNet.predict(140, 70);