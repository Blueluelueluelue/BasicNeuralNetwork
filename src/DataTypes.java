public class DataTypes {
    private static int[] printVals = {13, 15, 13, 9};
    private static String[] out = {"SUNNY", "OVERCAST", "RAINY"};
    private static String[] temp = {"HOW", "MILD", "COOL"};
    private static String[] hum = {"HIGH", "NORMAL"};
    private static String[] wind = {"WEAK", "STRONG"};

    class Outlook {
        public static final double SUNNY = 1.0;
        public static  final double OVERCAST = 2.0;
        public static final double RAINY = 3.0;
    }

    class Temp {
        public static final double HOT = 1.0;
        public static final double MILD = 2.0;
        public static final double COOL = 3.0;
    }

    class Humidity {
        public static final double HIGH = 1.0;
        public static final double NORMAL = 2.0;
    }

    class Wind {
        public static final double WEAK = 1.0;
        public static final double STRONG = 2.0;
    }


    public static void printTopRow() {

        String n = "#", o = "OUTLOOK", t = "TEMPERATURE", h = "HUMIDITY", w = "WIND", p = "PLAYTENNIS?";
        String test = "%3s";
        for (int i = 0; i < printVals.length; i++) {
            test += "%" + printVals[i] + "s";
        }
        test += "%13s";
        System.out.printf(test, n, o, t, h, w, p);
        System.out.println();
    }
    public static void printRow(int rnum, double... data) {
        String res = String.format("%3s", rnum + ".");
        String[] darr = new String[data.length];
        for (int i = 0; i < data.length; i++) {
            res += "%" + printVals[i] + "s";
            switch(i) {
                case 0:
                    darr[i] = out[(int)data[i]-1];
                    break;
                case 1:
                    darr[i] = temp[(int)data[i]-1];
                    break;
                case 2:
                    darr[i] = hum[(int)data[i]-1];
                    break;
                case 3:
                    darr[i] = wind[(int)data[i]-1];
                    break;
            }
        }
        System.out.printf(res, darr[0], darr[1], darr[2], darr[3]);
    }
    public static void printTennis(double[] guess) {
        System.out.printf("%12s\n", Double.max(guess[0], guess[1]) == guess[0] ? "YES" : "NO");
    }
}
