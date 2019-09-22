import { Utils } from './utils';

export class Neuron {
    public weights: number[];
    public bias: number;

    /**
     * @description constructor
     * 
     * @param weights 
     */
    constructor(weights: number[]) {
        this.bias = 0;
        this.weights = weights;
    }

    /**
     * @description calculate the feed forward which is weightX * inputX + bias
     * 
     * @param inputs 
     */
    public feedforward(inputs: number[]): number {
        let result: number = 0;
        for(let i = 0; i < this.weights.length; i++) {
            result += this.weights[i] * inputs[i];
        }
        return result += this.bias;
    }
}
