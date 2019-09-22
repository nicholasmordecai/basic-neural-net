import { Neuron } from './Neuron';
import { Utils } from './utils';
import { writeFileSync, readFileSync } from 'fs';

export class NeuralNet {
    private numberOfNeurons: number;
    private neurons: Neuron[];
    private trueData: number[];
    private trainingData: Array<{ weight: number, height: number }>;
    private neuron1: Neuron;
    private neuron2: Neuron;
    private outputNeuron: Neuron;

    constructor() {
        this.numberOfNeurons = 2;
        this.neurons = [];

        this.trainingData = JSON.parse(readFileSync(__dirname + '/../data/training_data.json', 'utf8'));
        this.trueData = JSON.parse(readFileSync(__dirname  + '/../data/true_results.json', 'utf8'));
        
        this.neuron1 = new Neuron([Utils.randomNormalDist(), Utils.randomNormalDist()]);
        this.neuron1.bias = Utils.randomNormalDist();
        this.neuron2 = new Neuron([Utils.randomNormalDist(), Utils.randomNormalDist()]);
        this.neuron2.bias = Utils.randomNormalDist();
        this.outputNeuron = new Neuron([Utils.randomNormalDist(), Utils.randomNormalDist()]);
        this.outputNeuron.bias = Utils.randomNormalDist();

        this.trainNetwork();

        const emily: number[] = [-7, -3];
        const result = this.feedForward(emily);
        console.log('Emily', result.toFixed(3));
    }

    private feedForward(x: number[]) {
        const result1: number = Utils.sigmoid(this.neuron1.weights[0] * x[0] + this.neuron1.weights[1] * x[1] + this.neuron1.bias);
        const result2: number = Utils.sigmoid(this.neuron2.weights[0] * x[0] + this.neuron2.weights[1] * x[1] + this.neuron2.bias);
        const output: number = Utils.sigmoid(this.outputNeuron.weights[0] * result1 + this.outputNeuron.weights[0] * result2 + this.outputNeuron.bias);
        return output;
    }

    private trainNetwork(): void {
        // Create neurons


        // Define learning parameters
        const learnRate: number = 0.1;
        const epochs: number = 1000;
        let epoch: number = 0;

        for (let i = 0; epoch < epochs; i++) {
            for (let i = 0; i < this.trainingData.length; i++) {
                const x = this.trainingData[i];
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

                // Update neurons
                this.neuron1.weights[0] -= learnRate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1;
                this.neuron1.weights[1] -= learnRate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2;
                this.neuron1.bias -= learnRate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1;

                this.neuron2.weights[0] -= learnRate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3;
                this.neuron2.weights[0] -= learnRate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4;
                this.neuron2.bias -= learnRate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2;

                this.outputNeuron.weights[0] -= learnRate * d_L_d_ypred * d_ypred_d_w5;
                this.outputNeuron.weights[0] -= learnRate * d_L_d_ypred * d_ypred_d_w6;
                this.outputNeuron.bias -= learnRate * d_L_d_ypred * d_ypred_d_b3;

                
            }
            if (epoch % 10 == 0) {
                const y_preds = this.applyAlongAxis();
                const loss: number = this.mseLoss(this.trueData, y_preds)
                console.log(`Epoch: ${epoch}, Loss: ${loss.toFixed(3)}`);
            }
            epoch++;
        }
    }

    private applyAlongAxis(): number[] {
        const result: number[] = [];
        for (let i = 0; i < this.trueData.length; i++) {
            result.push(this.feedForward([this.trainingData[i].weight, this.trainingData[i].height]));
        }
        return result;
    }

    /**
     * Mean Squared Error
     * @param trueValues
     * @param outputValues 
     */
    private mseLoss(trueValues: number[], outputValues: number[]): number {
        const diff = this.arrayDiff(trueValues, outputValues);
        return this.mean(diff);
    }

    private arrayDiff(array1: number[], array2: number[]): number[] {
        const diff: number[] = [];
        for (let i = 0; i < array1.length; i++) {
            diff.push(Math.pow(array1[i] - array2[i], 2));
        }
        return diff;
    }

    /**
     * Calculate the mean of a single dimensional array
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

new NeuralNet();