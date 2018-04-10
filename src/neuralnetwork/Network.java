package neuralnetwork;

import csv.CSV;
import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;
import trainset.TrainSet;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

import static csv.CSV.loadCSV;

public class Network {

    private double[][] outputs;
    private double[][][] weights;
    private double[][] bias;
    private double[][] errors;
    private double[][] outputs_derivative;
    private double learningRate;

    public final int[] NETWORK_LAYER_SIZE;
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int NETWORK_SIZE;

    public Network(int... NETWORK_LAYER_SIZE) {
        this.NETWORK_LAYER_SIZE = NETWORK_LAYER_SIZE;
        this.INPUT_SIZE = this.NETWORK_LAYER_SIZE[0];
        this.NETWORK_SIZE = this.NETWORK_LAYER_SIZE.length;
        this.OUTPUT_SIZE = this.NETWORK_LAYER_SIZE[this.NETWORK_SIZE-1];

        this.outputs = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];
        this.errors = new double[NETWORK_SIZE][];
        this.outputs_derivative = new double[NETWORK_SIZE][];

        this.learningRate = 0.3;
        double[] biasBounds = new double[]{-0.5, 0.7};
        double[] weightsBounds = new double[]{-1.0, 1.0};


        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.outputs[i] = new double[NETWORK_LAYER_SIZE[i]];
            this.errors[i] = new double[NETWORK_LAYER_SIZE[i]];
            this.outputs_derivative[i] = new double[NETWORK_LAYER_SIZE[i]];
            this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZE[i], biasBounds[0], biasBounds[1]);

            if(i > 0) {
                this.weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZE[i], NETWORK_LAYER_SIZE[i-1], weightsBounds[0], weightsBounds[1]);
            }
        }
    }

    public double[] calculate(double... inputs) {
        // return if the amount of inputs does not match the amount we are expecting
        if (inputs.length != INPUT_SIZE) {
            return null;
        }

        // Assign the first layer of the outputs
        outputs[0] = inputs;

        // Visit every layer
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            // Visit every neuron in that layer
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZE[layer]; neuron++) {

                // Calculate sum for each neuron

                // Since the bias has to be added anyway, we can initialize the sum using the bias
                double sum = bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZE[layer-1]; prevNeuron++) {
                    sum += outputs[layer-1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                outputs[layer][neuron] = sigmoid(sum);
                outputs_derivative[layer][neuron] = outputs[layer][neuron] * (1 - outputs[layer][neuron]);

            }
        }

        // returning the output at the last layer which is the output layer
        return outputs[NETWORK_SIZE-1];
    }

    private double sigmoid(double x) {
        return 1d / ( 1 + Math.exp(-x));
    }

    public void train(TrainSet set, int loops, int batchSize) {

        // If the input and target size are not what we expect then the data is not usable and we can't work with it
        if(set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE) return;

        // Repeat training procedure loops many times
        for(int i = 0; i < loops; i++) {
            TrainSet batch = set.extractBatch(batchSize);

            // Going through each element of the batch and using that element to train the network
            for(int b = 0; b < batch.size(); b++) {
                this.train(batch.getInput(b), batch.getOutput(b), learningRate);
            }

            // In case the user wants to see the mean squared error at each loop
            System.out.println(MSE(batch));
        }
    }

    private double MSE(double[] input, double[] target) {
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) {
            return 0.0;
        }

        // Populating the outputs[][] array so that it has values when we look inside it
        calculate(input);

        double sumE = 0;
        // Definition of MSE is take average of the squares of the error
        for (int i = 0; i < target.length; i++) {
            double e = target[i] - outputs[NETWORK_SIZE-1][i];
            sumE += e*e;
        }

        // Dividing by 2 * target.length because we used the derivative of output to calculate the error,
        // So when we are calculating MSE we integrate, but since we first squared it, the integration
        // Will be 1/2 times the expression
        return sumE / (2d * target.length);
    }

    private double MSE(TrainSet set) {
        double sumE = 0;
        for (int i = 0; i < set.size(); i++) {
            sumE += MSE(set.getInput(i), set.getOutput(i));
        }
        // The above logic does not apply for the MSE in a TrainSet since we are already taking that into account
        // When we call the MSE function inside the for loop.
        return sumE / set.size();
    }

    public void train(double[] inputs, double[] targets, double learningRate) {
        // Training won't work if the input size and target size are unexpected
        if (targets.length != OUTPUT_SIZE || inputs.length != INPUT_SIZE) {
            return;
        }
        // Populate the outputs[][]
        calculate(inputs);

        // Populate the errors[][]
        backpropError(targets);

        // Get new weights
        updateWeights(learningRate);
    }

    private void backpropError(double[] targets) {

        // Calculate errors for the output layer
        // Go through each neuron of the output layer
        for (int neuron = 0; neuron < NETWORK_LAYER_SIZE[NETWORK_SIZE-1]; neuron++) {
            errors[NETWORK_SIZE-1][neuron] = (outputs[NETWORK_SIZE-1][neuron] - targets[neuron])
                    * outputs_derivative[NETWORK_SIZE-1][neuron];
        }

        // Calculate errors for the neurons in the hidden layers
        for (int layer = NETWORK_SIZE - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZE[layer]; neuron++) {
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZE[layer+1]; nextNeuron++) {
                    sum += this.errors[layer+1][nextNeuron] * weights[layer+1][nextNeuron][neuron];
                }
                this.errors[layer][neuron] = outputs_derivative[layer][neuron] * sum;
            }
        }
    }

    private void updateWeights(double lr) {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZE[layer]; neuron++) {

                double delta = -1 * lr * errors[layer][neuron];
                bias[layer][neuron] += delta;

                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZE[layer - 1]; prevNeuron++) {
                    weights[layer][neuron][prevNeuron] += delta * outputs[layer-1][prevNeuron];
                }

            }
        }
    }

    public void saveNetwork(String fileName) throws Exception {
        Parser p = new Parser();
        p.create(fileName);
        Node root = p.getContent();
        Node netw = new Node("Network");
        Node ly = new Node("Layers");
        netw.addAttribute(new Attribute("sizes", Arrays.toString(this.NETWORK_LAYER_SIZE)));
        netw.addChild(ly);
        root.addChild(netw);
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {

            Node c = new Node("" + layer);
            ly.addChild(c);
            Node w = new Node("weights");
            Node b = new Node("biases");
            c.addChild(w);
            c.addChild(b);

            b.addAttribute("values", Arrays.toString(this.bias[layer]));

            for (int we = 0; we < this.weights[layer].length; we++) {

                w.addAttribute("" + we, Arrays.toString(weights[layer][we]));
            }
        }
        p.close();
    }

    public static Network loadNetwork(String fileName) throws Exception {

        Parser p = new Parser();

        p.load(fileName);
        String sizes = p.getValue(new String[] { "Network" }, "sizes");
        int[] si = ParserTools.parseIntArray(sizes);
        Network ne = new Network(si);

        for (int i = 1; i < ne.NETWORK_SIZE; i++) {
            String biases = p.getValue(new String[] { "Network", "Layers", i + "", "biases" }, "values");
            double[] bias = ParserTools.parseDoubleArray(biases);
            ne.bias[i] = bias;

            for(int n = 0; n < ne.NETWORK_LAYER_SIZE[i]; n++){

                String current = p.getValue(new String[] { "Network", "Layers", i + "", "weights" }, ""+n);
                double[] val = ParserTools.parseDoubleArray(current);

                ne.weights[i][n] = val;
            }
        }
        p.close();
        return ne;
    }

    public static void main(String[] args) {
        String abpath = new File("").getAbsolutePath();
        //String path = abpath + "/res/testCSV.csv";
        String path2 = abpath + "/res/SampleData.csv";
//        String path2 = abpath + "/res/test2CSV.csv";
        try {
            ArrayList<String[]> fileContent = loadCSV(path2);
            TrainSet set = CSV.parseCSV(fileContent);
//            Network network = new Network(set.INPUT_SIZE, 10, 9, 8, set.OUTPUT_SIZE);

//            CSV.createTestCSV(path2, 100);
            Network network = loadNetwork("res/NNFile");
//            TrainSet.trainData(network, set, 10, 1000, 30);
//            TrainSet.testTrainSet(network, set, 10);

            System.out.println(Arrays.toString(network.calculate(1.0, 1.0, 0.5, 0.5)));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
