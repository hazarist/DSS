import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class Controller {
	static final int PERCENT_SPLIT = 66;
	Instances[] ins;
	public Controller(BufferedReader dataset) throws Exception {
		Preprocessing preprocessing = new Preprocessing();
		ins = preprocessing.GetProccessedIns(dataset);
	}
	
	public ClassifierObject StartTrainAndTest() throws Exception {
		
		Instances NormalizedNumericIns = ins[0]; // NormalizedNumericIns
		Instances DiscretizedIns = ins[1]; // DiscretizedIns
		
		Instances[] numericTrainAndTestIns = splitDataset(NormalizedNumericIns);
		Instances[] nominalTrainAndTestIns = splitDataset(DiscretizedIns);
		
		Classifiers classifiers = new Classifiers();
		
		ClassifierObject[] allClassifiers = new ClassifierObject[5];
		
		allClassifiers[0] = new ClassifierObject(classifiers.NaiveBayes(nominalTrainAndTestIns[0]),"Naive Bayes");
		allClassifiers[0].setScore(Evaluater(allClassifiers[0].getCl(),nominalTrainAndTestIns[0],nominalTrainAndTestIns[1]));
		
		 allClassifiers[1] = new ClassifierObject(classifiers.DecisionTree(nominalTrainAndTestIns[0]),"Decision Tree");
		 allClassifiers[1].setScore(Evaluater(allClassifiers[1].getCl(),nominalTrainAndTestIns[0],nominalTrainAndTestIns[1]));
		 
		 allClassifiers[2] = new ClassifierObject(classifiers.KNearestNeighbour(3,numericTrainAndTestIns[0]),"K Nearest Neighbour");
		 allClassifiers[2].setScore(Evaluater(allClassifiers[2].getCl(),numericTrainAndTestIns[0],numericTrainAndTestIns[1]));
		
		 allClassifiers[3] = new ClassifierObject(classifiers.ArtificalNeuralNetwork(numericTrainAndTestIns[0]),"Artifical Neural Network");
		 allClassifiers[3].setScore(Evaluater(allClassifiers[3].getCl(),numericTrainAndTestIns[0],numericTrainAndTestIns[1]));
		
		 allClassifiers[4] = new ClassifierObject(classifiers.SupportVectorMachine(numericTrainAndTestIns[0]),"Support Vector Machine");
		 allClassifiers[4].setScore(Evaluater(allClassifiers[4].getCl(),numericTrainAndTestIns[0],numericTrainAndTestIns[1]));
		
		 int maxIndex = 0;
		 for(int i = 0; i < allClassifiers.length -1 ; i++) {
			 if(allClassifiers[i].getScore() < allClassifiers[i+1].getScore()) {
				maxIndex = i+1;	
			 }
		 }
		
		 
		 
		 return allClassifiers[maxIndex];
	}
	
	public void StartPrediction(ClassifierObject cl,BufferedReader dataset) throws Exception {
		Instances ins = new Instances(dataset);
		ins.setClassIndex(ins.numAttributes() - 1);
		
		Instances classifiedIns = Classify(cl.getCl(),ins);
		WriteLabeledInstances(classifiedIns);
	}
	
	public static double Evaluater(Classifier cl, Instances trainIns,Instances testIns) throws Exception {
		Evaluation eval = new Evaluation(trainIns);
		eval.evaluateModel(cl, testIns);
		System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		return eval.pctCorrect();
	}

	public static Instances[] splitDataset(Instances ins) {
		int trainSize = ins.numInstances() * PERCENT_SPLIT / 100;
		int testSize = ins.numInstances() - trainSize;
		
		Instances trainIns = new Instances(ins,0,trainSize);
		Instances testIns = new Instances(ins,trainSize,testSize);
		
		Instances[] trainAndTest = {trainIns, testIns};
		
		return trainAndTest;
	}
	
	public Instances Classify(Classifier cl,Instances ins) throws Exception {
		Instances labeled = new Instances(ins);
		for(int i = 0; i < ins.numInstances(); i++) {
			double clsLabel = cl.classifyInstance(ins.instance(i));
			labeled.instance(i).setClassValue(clsLabel);
		}
		
		return labeled;
	}
	
	public void WriteLabeledInstances(Instances ins) throws IOException {
		BufferedWriter writer = new BufferedWriter(
                new FileWriter("labeled.arff"));
		writer.write(ins.toString());
		writer.newLine();
		writer.flush();
		writer.close();
	}
}
