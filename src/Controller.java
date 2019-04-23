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
	
	public Classifier StartTrainAndTest() throws Exception {
		
		Instances NormalizedNumericIns = ins[0]; // NormalizedNumericIns
		Instances DiscretizedIns = ins[1]; // DiscretizedIns
		
		Instances[] numericTrainAndTestIns = splitDataset(NormalizedNumericIns);
		Instances[] nominalTrainAndTestIns = splitDataset(DiscretizedIns);
		
		Classifiers classifiers = new Classifiers();
		
		Classifier[] allClassifiers = new Classifier[5];
		double[] pctValues = new double[5];
		
		allClassifiers[0] = classifiers.NaiveBayes(nominalTrainAndTestIns[0]);
		pctValues[0] = Evaluater(allClassifiers[0],nominalTrainAndTestIns[0],nominalTrainAndTestIns[1]);
		//System.out.println(new DecimalFormat("###.###").format(pct1));
		
		 allClassifiers[1] = classifiers.DecisionTree(nominalTrainAndTestIns[0]);
		 pctValues[1] = Evaluater(allClassifiers[1],nominalTrainAndTestIns[0],nominalTrainAndTestIns[1]);
		
		 allClassifiers[2] = classifiers.KNearestNeighbour(3,numericTrainAndTestIns[0]);
		 pctValues[2] = Evaluater(allClassifiers[2],numericTrainAndTestIns[0],numericTrainAndTestIns[1]);
		
		 allClassifiers[3] = classifiers.ArtificalNeuralNetwork(numericTrainAndTestIns[0]);
		 pctValues[3] = Evaluater(allClassifiers[3],numericTrainAndTestIns[0],numericTrainAndTestIns[1]);
		
		 allClassifiers[4] = classifiers.SupportVectorMachine(numericTrainAndTestIns[0]);
		 pctValues[4] = Evaluater(allClassifiers[4],numericTrainAndTestIns[0],numericTrainAndTestIns[1]);
		
		 int maxIndex = 0;
		 for(int i = 0; i < pctValues.length-1 ; i++) {
			 if(pctValues[i] < pctValues[i+1]) {
				maxIndex = i+1;	
			 }
		 }
		
		 System.out.println("Bast algorithm perception: " + new DecimalFormat("###.###").format(pctValues[maxIndex]));
		 
		 return allClassifiers[maxIndex];
	}
	
	public void StartPrediction(Classifier cl,BufferedReader dataset) throws Exception {
		Instances ins = new Instances(dataset);
		ins.setClassIndex(ins.numAttributes() - 1);
		
		Instances classifiedIns = Classify(cl,ins);
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
