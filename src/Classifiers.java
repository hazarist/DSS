import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;

public class Classifiers{
	
	public Classifier NaiveBayes() {
		return new NaiveBayes();
	}
	
	public Classifier DecisionTree() {
		return new J48();
	}
	
	public Classifier KNearestNeighbour(int k) {
		return new IBk(k);
	}
	
	public Classifier KNearestNeighbour() {
		return new IBk();
	}
	
	public Classifier ArtificalNeuralNetwork() {
		return new MultilayerPerceptron();
	}
	
	public Classifier SupportVectorMachine() {
		return new SMO();
	}
	
}
