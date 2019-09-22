export class Utils {
    /**
     * @description Sigmoid function for calculating a sigmoid curve on 0 to 1
     * 
     * @param value 
     */
    public static sigmoid(value: number): number {
        return 1 / (1 + Math.pow(Math.E, -value));
    }

    /**
     * @description Derivative sigmoid function for calculating a derivative sigmoid curve on 0 to 1
     * 
     * @param value 
     */
    public static derivativeSigmoid(value: number): number {
        const fx: number = Utils.sigmoid(value);
        return fx * (1 - fx);
    }

    /**
     * @description generate a random float as a normal distribution number using a Box-Muller transform
     */
    public static randomNormalDist() {
        var u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    }
}