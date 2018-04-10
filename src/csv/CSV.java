package csv;

import neuralnetwork.NetworkTools;
import trainset.TrainSet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class CSV {

    public static void createTestCSV(String filePath, int rows) {
        double factor = 10.0;
        int rowSize = (int) (Math.random() * 7) + 3;
        int targetSize = (int) Math.log(rows) + 10;//(int) (Math.random() * (rows/100) - 2) + 2;
        double[] targetPool = new double[targetSize];

        for (int i = 0; i < targetSize; i++) {
            targetPool[i] = Math.random() * factor;
        }

        try {
            FileWriter wr = new FileWriter(filePath);
            for (int i = 0; i < rows; i++) {
                StringBuilder line = new StringBuilder();
                for (int j = 0; j < rowSize - 1; j++) {
                    line.append(String.format("%1.2f,", Math.random() * factor));
                }
                int index = (int) (Math.random() * targetSize);
                line.append(String.format("%1.2f\n", targetPool[index]));

                wr.write(line.toString());
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

    public static String[] removeRepeats(String[] cand) {
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

    public static double[] removeRepeats(double[] cand) {
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

    public static int linearSearch(double[] arr, double term) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == term) {
                return i;
            }
        }
        return -1;
    }

    public static int linearSearch(String[] arr, String term) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].equals(term)) {
                return i;
            }
        }
        return -1;
    }

    public static boolean areAllStrings(ArrayList<String[]> fileContent) {
        boolean are = false;
        for (String[] strarr: fileContent) {
            for (String s: strarr) {
                try {
                    double d = Double.parseDouble(s);
                    are = false;
                } catch (NumberFormatException e) {
                    are = true;
                }
            }
        }
        return are;
    }

    public static boolean isFirstRow(ArrayList<String[]> fileContent, boolean areStrings) {
        String[] firstRow = fileContent.get(0);
        boolean isFirst;
        if (areStrings) {
            String[] allRows = new String[(fileContent.size()-1) * fileContent.get(0).length];
            for (int i = 1; i < fileContent.size(); i++) {
                for (int j = 0; j < fileContent.get(i).length; j++) {
                    allRows[(i-1)*fileContent.get(i).length + j] = fileContent.get(i)[j];
                }
            }
            isFirst = true;
            for (String a : firstRow) {
                for (String b : allRows) {
                    if (a.equals(b)) {
                        isFirst = false;
                        break;
                    }
                }
                if (!isFirst)
                    break;
            }
        } else {
            ArrayList<String[]> temp = new ArrayList<>();
            temp.add(firstRow);
            isFirst = areAllStrings(temp);
        }
        return isFirst;
    }

    public static TrainSet parseCSV(ArrayList<String[]> fileContent) {
        double[][] fileDouble;

        for (String[] aFileContent : fileContent) {
            for (int j = 0; j < aFileContent.length; j++) {
                aFileContent[j] = aFileContent[j].toLowerCase().trim();
            }
        }

        if (areAllStrings(fileContent)) {
            if (isFirstRow(fileContent, true)) {
                fileContent.remove(0);
            }
            fileDouble = new double[fileContent.size()][fileContent.get(0).length];
            for (int j = 0; j < fileContent.get(0).length; j++) {
                String[] col = new String[fileContent.size()];
                for (int i = 0; i < fileContent.size(); i++) {
                    col[i] = fileContent.get(i)[j];
                }
                String[] distinctCol = removeRepeats(col);
                double[] distinctColNum = new double[distinctCol.length];
                for (int i = 0; i < distinctColNum.length; i++) {
                    distinctColNum[i] = i + 1;
                }
                for (int i = 0; i < col.length; i++) {
                    int index = linearSearch(distinctCol, col[i]);
                    fileDouble[i][j] = distinctColNum[index];
                }
            }
        } else {
            if (isFirstRow(fileContent, false)) {
                fileContent.remove(0);
            }
            fileDouble = new double[fileContent.size()][];
            for (int i = 0; i < fileDouble.length; i++) {
                fileDouble[i] = new double[fileContent.get(i).length];
                for (int j = 0; j < fileDouble[i].length; j++) {
                    fileDouble[i][j] = Double.parseDouble(fileContent.get(i)[j]);
                }
            }
        }
        return parseCSV(fileDouble);
    }


    public static TrainSet parseCSV(double[][] fileContent) {

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
