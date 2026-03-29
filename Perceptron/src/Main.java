import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {

    static double[] weights;
    static double theta;
    static Map<String, Integer> labeltoBin = new HashMap<>();
    static Map<Integer, String> bintoLabel = new HashMap<>();

    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);

    }


    //load data
    static void trainPerceptron(String trainFile, double alpha, int epochs) throws IOException {
        List<double[]> trainingFeatures = new ArrayList<>();
        List<Integer> exOutputs = new ArrayList<>();

        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String line;

        while ((line = br.readLine()) != null) {

            if (line.trim().isEmpty()) continue;

            String[] parts = line.split(",");
            double[] features = new double[parts.length - 1];
            for (int i = 0; i < parts.length - 1; i++) {
                features[i] = Double.parseDouble(parts[i].trim());

            }
            String stringLabel = parts[parts.length - 1].trim();
            if (labeltoBin.containsKey(stringLabel)) {
                int binVal = labeltoBin.size();
                if (binVal > 1) {
                    System.out.println("Perceptron only supports 2 classes. found extra class: " + stringLabel);

                }
                labeltoBin.put(stringLabel, binVal);
                bintoLabel.put(binVal, stringLabel);
            }

            trainingFeatures.add(features);
            exOutputs.add(labeltoBin.get(stringLabel));

        }
        br.close();

        if (trainingFeatures.isEmpty()) {
            System.out.println("Training file is empty.");
        }

        // Weights
        int numFeatures = trainingFeatures.get(0).length;
        weights = new double[numFeatures];
        Random rand = new Random();
        for (int i = 0; i < numFeatures; i++) {
            weights[i] = rand.nextDouble();
        }
        theta = rand.nextDouble();

        // training loop
        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < trainingFeatures.size(); i++) {
                double[] x = trainingFeatures.get(i);
                int d = exOutputs.get(i);

                double net = 0.0;

                for (int j = 0; j < d; j++) {
                    net += weights[j] * x[j];
                }

                net -= theta;

                int y = (net >= 0) ? 1 : 0;

                //delta update
                double error = d- y;
                for (int j = 0; j < weights.length; j++) {
                    weights[j] = weights[j] + alpha * error * x[j];
                }
                theta = theta - alpha * error;
            }
        }
        System.out.println("Training complete. Learned weights: " + Arrays.toString(weights) + " Bias: " + theta);


    }

    static String predict(double[] features) {
        double net = 0.0;
        for (int i = 0; i < weights.length; i++) {
            net += weights[i] * features[i];
        }
        net -= theta;
        int y = (net >= 0) ? 1 : 0;
        return bintoLabel.get(y);
    }

    static void testPerecepton(String trainFile) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String line;
        int correct = 0;
        int total = 0;
        while ((line = br.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            String[] parts = line.split(",");
            double[] features = new double[parts.length - 1];
            for (int i = 0; i < parts.length - 1; i++) {
                features[i] = Double.parseDouble(parts[i].trim());
            }

            String actualLabel = parts[parts.length - 1].trim();
            String predictedLabel = predict(features);

            System.out.println("Vector: " + Arrays.toString(features) + " | Expected: " + actualLabel + " | Predicted: " + predictedLabel);

            if (predictedLabel != null && predictedLabel.equals(actualLabel)) {
                correct++;
            }
            total++;

        }
        br.close();

        System.out.println("Correct: " + correct + " out of " + total);
        System.out.println("Accuracy: " + (double) correct / (double) total);


    }



}
