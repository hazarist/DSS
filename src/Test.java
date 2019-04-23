import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;


public class Test {
	
	static final int PERCENT_SPLIT = 66;
	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		String datasetPath = "train.arff";
		BufferedReader dataset = new BufferedReader(new FileReader(datasetPath));
		
		Preprocessing preprocessing = new Preprocessing();
		Instances[] ins = preprocessing.GetProccessedIns(dataset);
		
		dataset.close();
		
		Instances datasetIns = ins[0]; // binary
		
		int trainSize = datasetIns.numInstances() * PERCENT_SPLIT / 100;
		int testSize = datasetIns.numInstances() - trainSize;
		
		Instances trainIns = new Instances(datasetIns,0,trainSize);
		Instances testIns = new Instances(datasetIns,trainSize,testSize);
		
		Classifiers classifiers = new Classifiers();
		Classifier cl = classifiers.SupportVectorMachine();
		cl.buildClassifier(trainIns);
		double pct1 = Evaluater(cl,trainIns,testIns);
		System.out.println(new DecimalFormat("###.###").format(pct1));
		
		cl = classifiers.DecisionTree();
		cl.buildClassifier(trainIns);
		double pct2 = Evaluater(cl,trainIns,testIns);

		System.out.println(new DecimalFormat("###.###").format(pct2));
	}
	
	public static double Evaluater(Classifier cl, Instances trainIns,Instances testIns) throws Exception {
		Evaluation eval = new Evaluation(trainIns);
		eval.evaluateModel(cl, testIns);
		System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		return eval.pctCorrect();
	}

}
