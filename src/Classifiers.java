import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;

public class Classifiers{
	
	//dataset must be nominal
	public Classifier NaiveBayes(Instances ins) throws Exception {
		Classifier cl = new NaiveBayes();
		cl.buildClassifier(ins);
		
		return cl;
	}
	
	//data type is not important
	public Classifier DecisionTree(Instances ins) throws Exception{
		Classifier cl = new J48();
		cl.buildClassifier(ins);
		
		return cl;
	}
	
	//dataset must be numeric
	public Classifier KNearestNeighbour(int k,Instances ins) throws Exception{
		Classifier cl = new IBk(k);
		cl.buildClassifier(ins);
		
		return cl;
	}
	
	//dataset must be numeric
	public Classifier KNearestNeighbour(Instances ins) throws Exception {
		Classifier cl = new IBk();
		cl.buildClassifier(ins);
		
		return cl;
	}
	
	//dataset must be numeric
	public Classifier ArtificalNeuralNetwork(Instances ins) throws Exception {
		Classifier cl = new MultilayerPerceptron();
		cl.buildClassifier(ins);
		
		return cl;
	}
	
	//dataset must be numeric
	public Classifier SupportVectorMachine(Instances ins) throws Exception {
		Classifier cl = new SMO();
		cl.buildClassifier(ins);
	
		return cl;
	}
	
}
