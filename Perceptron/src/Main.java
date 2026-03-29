import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {

        Scanner sc = new Scanner(System.in);

        System.out.print("learning rate (alpha): ");
        double alpha = Double.parseDouble(sc.nextLine());

        System.out.print("repetitions: ");
        int repetitions = Integer.parseInt(sc.nextLine());

        System.out.print("(1) test your own vector. (2) test a set of vectors: ");
        String choice = sc.nextLine();

        if (choice.equals("1")) {
            System.out.print("Enter value to test your own vector: ");
            String[] parts = sc.nextLine().split(",");
            double[] testVector = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                testVector[i] = Double.parseDouble(parts[i].trim());
            }

            System.out.print("Provide a training file: ");
            String trainFile = sc.nextLine();

            String result = predictPerceptron(testVector, trainFile, alpha, repetitions);
            System.out.println("Predicted: " + result);

        } else if (choice.equals("2")) {
            System.out.print("Provide a file to test: ");
            String testFile = sc.nextLine();

            System.out.print("Provide a training file: ");
            String trainFile = sc.nextLine();

            BufferedReader testReader = new BufferedReader(new FileReader(testFile));

            int correct = 0;
            int total = 0;

            String line;
            while ((line = testReader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;

                String[] parts = line.split(",");
                if (parts.length < 2) {
                    continue;
                }

                double[] features = new double[parts.length - 1];
                for (int i = 0; i < parts.length - 1; i++) {
                    features[i] = Double.parseDouble(parts[i].trim());
                }

                String actualLabel = parts[parts.length - 1].trim();
                String predicted = predictPerceptron(features, trainFile, alpha, repetitions);

                System.out.println("\nchecking vector: " + Arrays.toString(parts));
                System.out.println("Predicted: " + predicted);

                if (predicted != null && predicted.equals(actualLabel)) {
                    System.out.println("Occurency");
                    correct++;
                }
                total++;
            }
            testReader.close();
            System.out.println("Correct answer: " + correct);
            System.out.println("Accuracy: " + (double) correct / (double) total);
        }
        sc.close();
    }

    static String predictPerceptron(double[] testVector, String trainFile, double alpha, int repetitions) throws Exception {

        BufferedReader reader = new BufferedReader(new FileReader(trainFile));

        List<double[]> trainData = new ArrayList<>();
        List<Integer> trainLabels = new ArrayList<>();
        Map<String, Integer> labelToBin = new HashMap<>();
        Map<Integer, String> binToLabel = new HashMap<>();

        String line;

        // 1. Read the training file
        while ((line = reader.readLine()) != null) {
            if (line.trim().isEmpty()) continue;

            String[] parts = line.split(",");
            if (parts.length < 2) continue;

            double[] features = new double[parts.length - 1];
            for (int i = 0; i < parts.length - 1; i++) {
                features[i] = Double.parseDouble(parts[i].trim());
            }

            String label = parts[parts.length - 1].trim();

            // Map text labels to 0 and 1
            if (!labelToBin.containsKey(label)) {
                int binValue = labelToBin.size();
                labelToBin.put(label, binValue);
                binToLabel.put(binValue, label);
            }

            trainData.add(features);
            trainLabels.add(labelToBin.get(label));
        }
        reader.close();

        if (trainData.isEmpty()) {
            return "Error: Training data empty";
        }

        // 2. Initialize Weights and Bias [0, 1]
        int numFeatures = trainData.get(0).length;
        double[] weights = new double[numFeatures];
        Random rand = new Random();
        for (int i = 0; i < numFeatures; i++) {
            weights[i] = rand.nextDouble();
        }
        double theta = rand.nextDouble();

        // 3. Train the Perceptron
        for (int e = 0; e < repetitions; e++) {
            for (int i = 0; i < trainData.size(); i++) {
                double[] x = trainData.get(i);
                int d = trainLabels.get(i);

                double net = 0.0;
                for (int j = 0; j < x.length; j++) {
                    net += weights[j] * x[j];
                }
                net -= theta;

                int y = (net >= 0) ? 1 : 0;
                double error = d - y;

                for (int j = 0; j < weights.length; j++) {
                    weights[j] += alpha * error * x[j];
                }
                theta -= alpha * error;
            }
        }

        // 4. Predict the result for the specific testVector passed to the method
        double net = 0.0;
        for (int i = 0; i < testVector.length; i++) {
            net += weights[i] * testVector[i];
        }
        net -= theta;

        int prediction = (net >= 0) ? 1 : 0;

        return binToLabel.get(prediction);
    }
}