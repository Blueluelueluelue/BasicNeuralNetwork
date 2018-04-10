package trainset;

import neuralnetwork.Network;
import neuralnetwork.NetworkTools;

import java.util.ArrayList;
import java.util.Arrays;

public class TrainSet {
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;

    private ArrayList<double[][]> data = new ArrayList<>();

    public TrainSet(int INPUT_SIZE, int OUTPUT_SIZE) {
        this.INPUT_SIZE = INPUT_SIZE;
        this.OUTPUT_SIZE = OUTPUT_SIZE;
    }

    public static void trainData(Network network, TrainSet set, int epochs, int loops, int batchSize) {
        for(int e = 0; e < epochs;e++) {
            network.train(set, loops, batchSize);
            System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>   "+ e + "   <<<<<<<<<<<<<<<<<<<<<<<<<<");
            try {
                network.saveNetwork("res/NNSaveFile");
            } catch (Exception e1) {
                e1.printStackTrace();
            }
        }
    }

    public static void testTrainSet(Network network, TrainSet set, int printSteps) {
        int correct = 0;
        for(int i = 0; i < set.size(); i++) {

            double highest = NetworkTools.indexOfHighestValue(network.calculate(set.getInput(i)));
            double actualHighest = NetworkTools.indexOfHighestValue(set.getOutput(i));
            if(highest == actualHighest) {
                correct ++ ;
            }

            if(i % printSteps == 0) {
                System.out.println(i + ": " + (double)correct / (double) (i + 1));
            }
        }
        System.out.println("Testing finished, RESULT: " + correct + " / " + set.size()+ "  -> " + (double)correct * 100/ (double)set.size() +" %");
    }

    public void addData(double[] in, double[] expected) {
        if (in.length != INPUT_SIZE || expected.length != OUTPUT_SIZE) {
            return;
        }
        data.add(new double[][]{in, expected});
    }

    public TrainSet extractBatch(int size) {
        if(size > 0 && size <= this.size()) {
            TrainSet set = new TrainSet(INPUT_SIZE, OUTPUT_SIZE);
            Integer[] ids = NetworkTools.randomValues(size,0, this.size() - 1);
            if (ids != null) {
                for(Integer i: ids) {
                    set.addData(this.getInput(i), this.getOutput(i));
                }
            }
            return set;
        } else {
            return this;
        }
    }

    public String toString() {
        StringBuilder s = new StringBuilder("TrainSet [" + INPUT_SIZE + " ; " + OUTPUT_SIZE + "]\n");
        int index = 0;
        for(double[][] r:data) {
            s.append(index).append(":   ").append(Arrays.toString(r[0])).append("  >-||-<  ").append(Arrays.toString(r[1])).append("\n");
            index++;
        }
        return s.toString();
    }

    public int size() {
        return data.size();
    }

    public double[] getInput(int index) {
        if(index >= 0 && index < size())
            return data.get(index)[0];
        else return null;
    }

    public double[] getOutput(int index) {
        if(index >= 0 && index < size())
            return data.get(index)[1];
        else return null;
    }

    public static void main(String[] args) {
        TrainSet set = new TrainSet(3,2);

        for(int i = 0; i < 8; i++) {
            double[] a = new double[3];
            double[] b = new double[2];
            for(int k = 0; k < 3; k++) {
                a[k] = Double.parseDouble(String.format("%1.1f", Math.random()));
                if(k < 2) {
                    b[k] = Double.parseDouble(String.format("%1.1f", Math.random()));
                }
            }
            set.addData(a,b);
        }

        System.out.println(set);
        System.out.println(set.extractBatch(3));
    }
}
