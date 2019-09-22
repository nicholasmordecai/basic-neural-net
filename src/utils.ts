export class Utils {
    public static rn(): number {
        return Math.floor(Math.random() * Math.floor(1000)) / 1000;
    }

    public static ri(): number {
        return Math.floor(Math.random() * Math.floor(1000));
    }

    public static sigmoid(value: number): number {
        return 1 / (1 + Math.pow(Math.E, -value));
    }

    public static derivativeSigmoid(value: number): number {
        const fx: number = Utils.sigmoid(value);
        return fx * (1 - fx);
    }

    public static randomNormalDist() {
        var u = 0, v = 0;
        while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
        while(v === 0) v = Math.random();
        return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    }
}