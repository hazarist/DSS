import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;

import weka.core.Instances;

public class Test {
	public static void main(String[] args) throws Exception {
	
		String datasetPath = "train.arff";
		String predictionDatasetPath = "prediction.arff";
		
		BufferedReader dataset = new BufferedReader(new FileReader(datasetPath));
		
		Controller cont = new Controller(dataset);
		ClassifierObject cl = cont.StartTrainAndTest();
		System.out.println("Best algorithm is " + cl.getName() + "\nBest algorithm perception: " + new DecimalFormat("###.###").format(cl.getScore()));
		dataset.close();
		
		BufferedReader predictionDataset = new BufferedReader(new FileReader(predictionDatasetPath));
		Instances ins = new Instances(predictionDataset);
		ins.setClassIndex(ins.numAttributes() - 1);
		cont.StartPrediction(cl, ins);
		predictionDataset.close();
		
	}
}
