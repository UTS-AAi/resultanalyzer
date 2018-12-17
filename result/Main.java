package result;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;

import java.io.*;

public class Main {

    private Classifier loadModel(String fileName) {

        Classifier model = null;

        try {
             model =  (Classifier) SerializationHelper.read(new FileInputStream(fileName));
        } catch (java.io.FileNotFoundException e) {
            System.out.println("Cannot find the file: " + fileName);
        } catch (java.lang.Exception e) {
            System.out.println("Cannot open the file: " + fileName);
            e.printStackTrace();
        }

        return model;
    }

    private Evaluation evaluateModel(Instances testData, Classifier model) {

        Evaluation eval = null;

        try {
          eval = new Evaluation(testData);
          eval.evaluateModel(model, testData);
        } catch (Exception e) {
            System.out.println("Cannot evaluate the moel.");
            e.printStackTrace();
        }

        return eval;
    }

    /***
     * The method runs all on all the datasets in given folders.
     *
     * @param args the first parameter is model folder, and the second parameter is
     *             the data folder. The model folder names must end with F[0-9],
     *             while data names have to have the following format:
     *             tst_" + fold + ".arff for test datasets
     */
    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.print("Incorrect number of parameters. Two folder names needed.");
            return;
        }

        try {
            Main resultAnalzer = new Main();
            File modelFolder = new File(args[0]);
            File dataFolder = new File(args[1]);

            Evaluation [] evaluations = new Evaluation[10];
            int [] positiveIndeces = new int[10];
            String [] selectedModels = new String [10];

            for (File file : modelFolder.listFiles()) {
                if (file.getName().matches(".*F[0-9]$")) {

                    String fold = file.getName().substring(file.getName().length() - 1);
                    System.out.println("Processing fold: " + fold);
                    System.out.println(file.getName());

                    Classifier model = resultAnalzer.loadModel(modelFolder + "/" + file.getName() +
                            "/trained.0.model");

                    if (model == null)
                        continue;
                    
                    BufferedReader reader =
                            new BufferedReader(new FileReader
                                    (dataFolder + "/tst_" + fold + ".arff"));
                    ArffLoader.ArffReader loader = new ArffLoader.ArffReader(reader);
                    Instances testData = loader.getData();
                    testData.setClassIndex(testData.numAttributes() - 1);

                    Evaluation evaluation = resultAnalzer.evaluateModel(testData, model);

                    System.out.println(evaluation.toSummaryString());

                    selectedModels[Integer.parseInt(fold)] =
                            model.toString().substring(0, model.toString().indexOf('\n'));
                    positiveIndeces[Integer.parseInt(fold)] = testData.classAttribute().indexOfValue("1");
                    evaluations[Integer.parseInt(fold)] = evaluation;
                }

            }

            for (int i = 0; i < 10; ++i) {
                Evaluation evaluation = evaluations[i];

                System.out.println(evaluation.pctCorrect() + ", " +
                        evaluation.precision(positiveIndeces[i]) + ", " +
                        evaluation.recall(positiveIndeces[i])  + ", " +
                        //evaluation.truePositiveRate(positiveIndeces[i]) + "," + = same as recall
                        evaluation.trueNegativeRate(positiveIndeces[i]) + "," +
                        evaluation.weightedFMeasure() + "," +
                        evaluation.areaUnderROC(positiveIndeces[i]) + "," +
                        evaluation.numTruePositives(positiveIndeces[i]) + "," +
                        evaluation.numTrueNegatives(positiveIndeces[i]) + "," +
                        evaluation.numFalsePositives(positiveIndeces[i]) + "," +
                        evaluation.numFalseNegatives(positiveIndeces[i]) + "," +
                        selectedModels[i]);
            }
        }
        catch (Exception e) {
            System.out.println("Error in main method: ");
            e.printStackTrace();
        }
    }
}
