package neuralnetwork;

import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;
import trainset.TrainSet;
import mnist.Mnist;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class Network {

    private double[][] outputs;
    private double[][][] weights;
    private double[][] bias;
    private double[][] errors;
    private double[][] outputs_derivative;
    private double learningRate;
    private double[] biasBounds;
    private double[] weightsBounds;

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
        this.biasBounds = new double[]{-0.5, 0.7};
        this.weightsBounds = new double[]{-1.0, 1.0};


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
        if(set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE) return;
        for(int i = 0; i < loops; i++) {
            TrainSet batch = set.extractBatch(batchSize);
            for(int b = 0; b < batchSize; b++) {
                this.train(batch.getInput(b), batch.getOutput(b), learningRate);
            }
            System.out.println(MSE(batch));
        }
    }

    public double MSE(double[] input, double[] target) {
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) {
            return 0.0;
        }
        calculate(input);
        double sumE = 0;
        for (int i = 0; i < target.length; i++) {
            double e = target[i] - outputs[NETWORK_SIZE-1][i];
            sumE += e*e;
        }
        return sumE / (2d * target.length);
    }

    public double MSE(TrainSet set) {
        double sumE = 0;
        for (int i = 0; i < set.size(); i++) {
            sumE += MSE(set.getInput(i), set.getOutput(i));
        }
        return sumE / set.size();
    }

    public void train(double[] inputs, double[] targets, double learningRate) {
        if (targets.length != OUTPUT_SIZE || inputs.length != INPUT_SIZE) {
            return;
        }
        calculate(inputs);
        backpropError(targets);
        updateWeights(learningRate);
    }

    public void backpropError(double[] targets) {

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

    public void updateWeights(double lr) {
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

    public static void createTestCSV(String filePath, int rows) {
        int rowSize = (int) (Math.random() * 7) + 3;
        int targetSize = (int) Math.log(rows) + 10;//(int) (Math.random() * (rows/100) - 2) + 2;
        double[] targetPool = new double[targetSize];

        for (int i = 0; i < targetSize; i++) {
            targetPool[i] = Math.random();
        }

        try {
            FileWriter wr = new FileWriter(filePath);
            for (int i = 0; i < rows; i++) {
                String line = "";
                for (int j = 0; j < rowSize - 1; j++) {
                    line += String.format("%1.2f,", Math.random());
                }
                int index = (int) (Math.random() * targetSize);
                line += String.format("%1.2f\n", targetPool[index]);

                wr.write(line);
            }
            wr.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static ArrayList<String[]> loadCSV(String filePath) {
        String splitBy = ",";
        String line;
        ArrayList<String[]> fileContent = new ArrayList<>();
        try {

            BufferedReader br = new BufferedReader(new FileReader(filePath));

            while ((line = br.readLine()) != null) {
                String[] row = line.split(splitBy);
                fileContent.addAll(Collections.singleton(row));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return fileContent;
    }

    public static double[] removeRepeats(double[] cand) {
        ArrayList<Double> temp = new ArrayList<>();
        for (int i = 0; i < cand.length; i++) {
            if(!temp.contains(cand[i])) {
                temp.add(cand[i]);
            }
        }
        double[] res = new double[temp.size()];
        for (int i = 0; i < res.length; i++) {
            res[i] = temp.get(i);
        }
        return res;
    }

    public static int linearSearch(double[] arr, double term) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == term) {
                return i;
            }
        }
        return -1;
    }

    public static TrainSet parseCSV(ArrayList<String[]> fileContent) {

        int targetSize;
        double[] targetCands = new double[fileContent.size()];

        for (int i = 0; i < targetCands.length; i++) {
            String[] s = fileContent.get(i);
            try {
                targetCands[i] = Double.parseDouble(s[s.length-1]);
            } catch (NumberFormatException e) {
                System.out.println("File contains stuff which are not numbers");
                e.printStackTrace();
                return null;
            }
        }

        double[] distinctTargets = removeRepeats(targetCands);
        targetSize = distinctTargets.length;
        TrainSet set = new TrainSet(fileContent.get(0).length - 1, targetSize);

        try {
            for (int i = 0; i < fileContent.size(); i++) {
                String[] s = fileContent.get(i);
                double t = Double.parseDouble(s[s.length-1]);
                double[] inputs = new double[s.length-1];
                for (int j = 0; j < inputs.length; j++) {
                    inputs[j] = Double.parseDouble(s[j]);
                }
                double[] targets = new double[distinctTargets.length];
                int index = linearSearch(distinctTargets, t);
                if (index != -1) {
                    targets[index] = 1d;
                }

                set.addData(inputs, targets);
            }
        } catch (NumberFormatException e) {
            System.out.println("File contains stuff which are not numbers");
            e.printStackTrace();
            return null;
        }

        return set;
    }


    public static void main(String[] args) {
        String path = new File("").getAbsolutePath() + "/res/testCSV.csv";
        String path2 = new File("").getAbsolutePath() + "/res/test2CSV.csv";

        try {
            ArrayList<String[]> fileContent = loadCSV(path2);
            for (String[] strarr : fileContent) {
                for (String s : strarr) {
                    System.out.print(s + " ");
                }
                System.out.println();
            }
            TrainSet set = parseCSV(fileContent);
            if (set != null) {
                int in = set.INPUT_SIZE;
                int out = set.OUTPUT_SIZE;
                Network network = new Network(in, in - 1, in - 2, in / 2, out);
                Mnist.trainData(network, set, 10, 500, 1000);
                Mnist.testTrainSet(network, set.extractBatch(200), 10);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

//        createTestCSV(path2, 20000);

        /*for (String[] strarr : fileContent) {
            for (String s : strarr) {
                System.out.print(s + " ");
            }
            System.out.println();
        }*/



        /*Network network = new Network(4, 3, 3, 2);




        System.out.println("--------------------------TRAINING DATA--------------------------");
        DataTypes.printTopRow();

        for (int i = 0; i < inputs.length; i++) {
            DataTypes.printRow(i + 1, inputs[i]);
            DataTypes.printTennis(network.calculate(inputs[i]));
        }
        System.out.println("--------------------------UNKNOWN DATA--------------------------");

        for (int i = 0; i < newData.length; i++) {
            DataTypes.printRow(i + 1, newData[i]);
            DataTypes.printTennis(network.calculate(newData[i]));
        }*/
        //Network net = new Network(4,3,3,2);

        /*TrainSet set = new TrainSet(4, 2);
        set.addData(new double[]{0.1,0.2,0.3,0.4}, new double[]{0.9,0.1});
        set.addData(new double[]{0.9,0.8,0.7,0.6}, new double[]{0.1,0.9});
        set.addData(new double[]{0.3,0.8,0.1,0.4}, new double[]{0.3,0.7});
        set.addData(new double[]{0.9,0.8,0.1,0.2}, new double[]{0.7,0.3});

        net.train(set, 10000, 4);

        for(int i = 0; i < 4; i++) {
            System.out.println(Arrays.toString(net.calculate(set.getInput(i))) + net.MSE(set.getInput(i), set.getOutput(i)));
        }
        System.out.println(net.MSE(set));*/
        /*for(int i = 0; i < 4; i++) {
            System.out.println(Arrays.toString(net.calculate(set.getInput(i))));
        }*/

        /*Network network = new Network(4, 3, 3, 2);
        try {
            network.saveNetwork("res/NNFile");
        } catch (Exception e) {
            e.printStackTrace();
        }*/
    }
}
