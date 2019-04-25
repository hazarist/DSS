import weka.classifiers.Classifier;

public class ClassifierObject {

	private Classifier cl;
	private String name;
	private double score;
	public ClassifierObject(Classifier cl, String name) {
		this.cl = cl;
		this.name = name;
	}

	public Classifier getCl() {
		return cl;
	}

	public String getName() {
		return name;
	}
	
	public void setScore(double score) {
		this.score = score;
	}
	
	public double getScore() {
		return score;
	}

}
