package neuralnetwork;


public class NetworkTools {

    public static double[] normalize(double[] array) {
        double[] arr = new double[array.length];
        double max = array[0];
        for (double d : array) {
            max = d > max ? d : max;
        }
        for (int i = 0; i < arr.length; i++) {
            arr[i] = array[i] / max;
        }
        return arr;
    }

    public static double[][] normalizeCols(double[][] array) {
        double[][] arr = new double[array.length][];
        for (int i = 0; i < array.length; i++) {
            arr[i] = new double[array[i].length];
        }

        for (int j = 0; j < array[0].length; j++) {
            double colMax = array[0][j];
            for (double[] anArray : array) {
                colMax = anArray[j] > colMax ? anArray[j] : colMax;
            }
            for (int i = 0; i < array.length; i++) {
                String temp = String.format("%1.2f", array[i][j] / colMax);
                arr[i][j] = Double.parseDouble(temp);
            }
        }
        return arr;
    }

    public static double[] createRandomArray(int size, double lowerBound, double upperBound) {
        if (size < 1) {
            return null;
        }

        double[] arr = new double[size];

        for (int i = 0; i < size; i++) {
            arr[i] = randomNumber(lowerBound, upperBound);
        }
        return arr;
    }

    public static double[][] createRandomArray(int sizeX, int sizeY, double lowerBound, double upperBound) {
        if (sizeX < 1 || sizeY < 1) {
            return null;
        }

        double[][] arr = new double[sizeX][sizeY];

        for (int i = 0; i < sizeX; i++) {
            arr[i] = createRandomArray(sizeY, lowerBound, upperBound);
        }
        return arr;
    }

    private static double randomNumber(double lower, double upper) {
        return Math.random()*(upper - lower) + lower;
    }

    public static Integer[] randomValues(int size, int lower, int upper) {

        upper++;
        if (size > (upper-lower)) {
            return null;
        }

        Integer[] arr = new Integer[size];

        for (int i = 0; i < size; i++) {
            int n = (int)randomNumber(lower, upper);
            while (containsValue(arr, n)) {
                n = (int)randomNumber(lower, upper);
            }
            arr[i] = n;
        }
        return arr;

    }

    public static <T extends Comparable<T>> boolean containsValue(T[] arr, T value) {
        for (T anArr : arr) {
            if (anArr != null) {
                if (value.compareTo(anArr) == 0) {
                    return true;
                }
            }
        }
        return false;
    }

    public static int indexOfHighestValue(double[] arr) {
        int index = 0;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > arr[index]) {
                index = i;
            }
        }
        return index;
    }
}
