import java.io.BufferedReader;
import java.io.File;
import java.io.Reader;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Normalize;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Discretize;

public class Preprocessing {

	public Instances[] GetProccessedIns(Reader dataset) throws Exception {
		
		Instances ins = new Instances(dataset);
		
		ins.setClassIndex(ins.numAttributes() - 1);
		
		ins = NormalizeIns(ins);
		ins = RandomizeIns(ins);
		Instances BinaryIns = NominalToBinary(ins);
		Instances DiscretizeIns = Discretize(ins);
		
		Instances[] insArray = {BinaryIns,DiscretizeIns};
		
		return insArray;
	}
	
	public Instances NormalizeIns(Instances ins) throws Exception {
		
		Filter normalizedIns = new Normalize();
		normalizedIns.setInputFormat(ins);
		ins = Filter.useFilter(ins, normalizedIns);
		
		return ins;
	}
	
	public Instances RandomizeIns(Instances ins) throws Exception{
		Filter randomizeIns = new Randomize();
		randomizeIns.setInputFormat(ins);
		ins = Filter.useFilter(ins, randomizeIns);
		
		return ins;
	}
	
	public Instances NominalToBinary(Instances ins) throws Exception{
		NominalToBinary ntb = new NominalToBinary();
		ntb.setInputFormat(ins);
		ins = Filter.useFilter(ins, ntb);
		
		return ins;
	}
	
	public Instances Discretize(Instances ins) throws Exception{
		Discretize disc = new Discretize();
		disc.setInputFormat(ins);
		ins = Filter.useFilter(ins, disc);
		
		return ins;
	}
}
