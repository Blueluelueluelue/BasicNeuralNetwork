package neuralnetwork;

public class NetworkTools {

    public static double[] createArray(int size, double initValue) {
        if (size < 1) {
            return null;
        }

        double[] arr = new double[size];

        for (int i = 0; i < size; i++) {
            arr[i] = initValue;
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
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] != null) {
                if (value.compareTo(arr[i]) == 0) {
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
