package csv;


import neuralnetwork.Network;
import neuralnetwork.NetworkTools;
import trainset.TrainSet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;

public class CSV {

    private final TrainSet trainSet;
    private String[] distinctTargets;
    private String[] topRow;
    private ArrayList<Hashtable<String, Double>> inputsTable;

    public TrainSet getTrainSet() {
        return trainSet;
    }

    public String[] getTopRow() {
        return topRow;
    }

    public CSV(String filePath) {
        inputsTable = new ArrayList<>();
        topRow = null;
        trainSet = loadCSV(filePath);
    }

    public String[] evaluate(Network network, String[] inputs) {
        double[] doubleInputs = new double[inputs.length];
        ArrayList<String[]> in = new ArrayList<>();
        for (String s : inputs) {
            String[] strArr = new String[1];
            strArr[0] = s;
            in.add(strArr);
        }
        if (areAllStrings(in)) {
            for (int i = 0; i < inputs.length ; i++) {
                doubleInputs[i] = inputsTable.get(i).get(inputs[i]);
            }
        } else {
            for (int i = 0; i < inputs.length; i++) {
                try {
                    doubleInputs[i] = Double.parseDouble(inputs[i]);
                } catch (NumberFormatException e) {
                    e.printStackTrace();
                }
            }
        }
        double[] guess = network.calculate(doubleInputs);
        String[] result = new String[inputs.length + 1];
        result[result.length - 1] = distinctTargets[NetworkTools.indexOfHighestValue(guess)];
        System.arraycopy(inputs, 0, result, 0, inputs.length);
        return result;
    }


    public static void createTestCSV(String filePath, int rows) {
        // factor defines how big the numbers will be in the csv file, factor of 10 will mean the numbers
        // Are between 0 and 10
        double factor = 10.0;

        // rowSize defines how many numbers there will be in a particular row, rowSize of 10 means
        // There will be 10 numbers in each row
        int rowSize = (int) (Math.random() * 7) + 3;

        // targetSize defines the number of possible targets there are
        int targetSize = (int) Math.log(rows) + 10;
        double[] targetPool = new double[targetSize];

        for (int i = 0; i < targetSize; i++) {
            // The likelihood of a target repeating is extremely so we don't care
            targetPool[i] = Math.random() * factor;
        }

        try {
            FileWriter wr = new FileWriter(filePath);
            for (int i = 0; i < rows; i++) {
                StringBuilder line = new StringBuilder();
                for (int j = 0; j < rowSize - 1; j++) {
                    // Numbers with more than 2 digits after decimal look weird
                    line.append(String.format("%1.2f,", Math.random() * factor));
                }

                // Each row gets a random target
                int index = (int) (Math.random() * targetSize);
                line.append(String.format("%1.2f\n", targetPool[index]));

                // Since line is StringBuilder and not a String
                wr.write(line.toString());
            }
            wr.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private TrainSet loadCSV(String filePath) {
        String splitBy = ",";
        String line;

        // An ArrayList for flexibility
        ArrayList<String[]> fileContent = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(filePath));

            while ((line = br.readLine()) != null) {
                String[] row = line.split(splitBy);

                // Since I want to maintain the String[] condition of each row
                fileContent.add(row);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return parseCSV(fileContent);
    }

    private static String[] removeRepeats(String[] cand) {
        ArrayList<String> temp = new ArrayList<>();
        for (String aCand : cand) {
            if (!temp.contains(aCand)) {
                temp.add(aCand);
            }
        }
        String[] res = new String[temp.size()];
        for (int i = 0; i < res.length; i++) {
            res[i] = temp.get(i);
        }
        return res;
    }

    private static double[] removeRepeats(double[] cand) {
        ArrayList<Double> temp = new ArrayList<>();
        for (double aCand : cand) {
            if (!temp.contains(aCand)) {
                temp.add(aCand);
            }
        }
        double[] res = new double[temp.size()];
        for (int i = 0; i < res.length; i++) {
            res[i] = temp.get(i);
        }
        return res;
    }

    private static int linearSearch(double[] arr, double term) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == term) {
                return i;
            }
        }
        return -1;
    }

    private static int linearSearch(String[] arr, String term) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].equals(term)) {
                return i;
            }
        }
        return -1;
    }

    private static boolean areAllStrings(ArrayList<String[]> fileContent) {
        boolean are = true;
        for (String[] strArr: fileContent) {
            for (String s: strArr) {
                try {
                    double d = Double.parseDouble(s);
                    are = false;
                } catch (NumberFormatException e) {
                    // If I encountered even a single non-double then the content isn't "All Strings"
                    return true;
                }
            }
        }
        return are;
    }

    private static boolean isFirstRow(ArrayList<String[]> fileContent, boolean areStrings) {
        String[] firstRow = fileContent.get(0);
        boolean isFirst;
        if (areStrings) {
            // Get all the rows of the content except the first row
            String[] allRows = new String[(fileContent.size()-1) * fileContent.get(0).length];
            for (int i = 1; i < fileContent.size(); i++) {
                for (int j = 0; j < fileContent.get(i).length; j++) {
                    allRows[(i-1)*fileContent.get(i).length + j] = fileContent.get(i)[j];
                }
            }

            // Assume it is the first row
            isFirst = true;

            // If I find any element inside the rest of the content which is equal to an element
            // Inside the first row then the first row is not the "First Row"
            for (String a : firstRow) {
                for (String b : allRows) {
                    if (a.equals(b)) {
                        isFirst = false;
                        break;
                    }
                }

                // No need to continue looking if even a single element satisfies the condition
                if (!isFirst)
                    break;
            }
        } else {
            // Making this ArrayList since the areAllStrings method expects an ArrayList<String[]>
            ArrayList<String[]> temp = new ArrayList<>();
            temp.add(firstRow);

            /* If the whole content isn't made up strings then only 2 possibilities are there
             Either the whole content is made of double values
             Or some of it is made of Strings but not all */

            // So if the first row is made of all strings then it has to be the "First Row" else not
            isFirst = areAllStrings(temp);
        }
        return isFirst;
    }

    private TrainSet parseCSV(ArrayList<String[]> fileContent) {
        double[][] fileDouble;

        for (String[] aFileContent : fileContent) {
            for (int j = 0; j < aFileContent.length; j++) {
                aFileContent[j] = aFileContent[j].toLowerCase().trim();
            }
        }

        if (areAllStrings(fileContent)) {
            if (isFirstRow(fileContent, true)) {
                topRow = fileContent.remove(0);
            }

            fileDouble = new double[fileContent.size()][fileContent.get(0).length];

            for (int j = 0; j < fileContent.get(0).length; j++) {
                String[] col = new String[fileContent.size()];
                Hashtable<String, Double> hashtable = new Hashtable<>();

                for (int i = 0; i < fileContent.size(); i++) {
                    col[i] = fileContent.get(i)[j];
                }
                String[] distinctCol = removeRepeats(col);

                // Populating the global distinctTargets array
                if (j == fileContent.get(0).length - 1) {
                    this.distinctTargets = new String[distinctCol.length];
                    System.arraycopy(distinctCol, 0, this.distinctTargets, 0, distinctCol.length);
                }

                double[] distinctColNum = new double[distinctCol.length];
                for (int i = 0; i < distinctColNum.length; i++) {
                    distinctColNum[i] = i + 1;
                }
                for (int i = 0; i < col.length; i++) {
                    int index = linearSearch(distinctCol, col[i]);
                    fileDouble[i][j] = distinctColNum[index];
                }

                if (j == fileContent.get(0).length - 1)
                    System.out.println("STOP");

                if (j < fileContent.get(0).length - 1) {
                    distinctColNum = NetworkTools.normalize(distinctColNum);
                    for (int i = 0; i < distinctColNum.length; i++) {
                        hashtable.put(distinctCol[i], distinctColNum[i]);
                    }
                    inputsTable.add(hashtable);
                }
            }
        } else {
            if (isFirstRow(fileContent, false)) {
                topRow = fileContent.remove(0);
            }
            fileDouble = new double[fileContent.size()][];
            String[] targetPool = new String[fileContent.size()];

            for (int i = 0; i < fileDouble.length; i++) {
                fileDouble[i] = new double[fileContent.get(i).length];
                targetPool[i] = String.valueOf(fileContent.get(i)[fileContent.get(i).length - 1]);
                for (int j = 0; j < fileDouble[i].length; j++) {
                    fileDouble[i][j] = Double.parseDouble(fileContent.get(i)[j]);
                }
            }

            // Populating the global distinctTargets array
            this.distinctTargets = removeRepeats(targetPool);
        }
        return parseCSV(fileDouble);
    }


    private TrainSet parseCSV(double[][] fileContent) {

        fileContent = NetworkTools.normalizeCols(fileContent);
        int targetSize;
        double[] targetCands = new double[fileContent.length];

        for (int i = 0; i < targetCands.length; i++) {
            double[] d = fileContent[i];
            targetCands[i] = d[d.length-1];
        }

        double[] distinctTargets = removeRepeats(targetCands);

        targetSize = distinctTargets.length;
        TrainSet set = new TrainSet(fileContent[0].length - 1, targetSize);

        for (double[] d : fileContent) {
            double t = d[d.length - 1];
            double[] inputs = new double[d.length - 1];
            System.arraycopy(d, 0, inputs, 0, inputs.length);
            double[] targets = new double[distinctTargets.length];
            int index = linearSearch(distinctTargets, t);
            if (index != -1) {
                targets[index] = 1d;
            }
            set.addData(inputs, targets);
        }
        return set;
    }
}
