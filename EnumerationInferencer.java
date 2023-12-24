//Developed by Hesham Elshafey and Ahmed Nassar
package bn.inference;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import bn.core.Assignment;
import bn.core.BayesianNetwork;
import bn.base.Distribution;
import bn.core.Range;
import bn.core.Inferencer;
import bn.core.RandomVariable;
import bn.core.Value;

public class EnumerationInferencer implements Inferencer {

	@Override
	public Distribution query(RandomVariable X, Assignment e, BayesianNetwork network) {
		return Ennumerationask(X, e, network);
	}

	static Distribution Ennumerationask(RandomVariable X, Assignment e, BayesianNetwork bn) {
		Distribution dist = new Distribution(X);
		List<RandomVariable> vars = bn.getVariablesSortedTopologically();
		Range x_range = X.getRange();
		Iterator itr = x_range.iterator();
		while (itr.hasNext()) {
			Value value = (Value) itr.next();
			e.put(X, value);
			double x_probablity = Ennumerateall(vars, e, bn);
			dist.set(value, x_probablity);
		}
		return dist;
	}

	static double Ennumerateall(List<RandomVariable> vars, Assignment e, BayesianNetwork bn) {
		List<RandomVariable> vars_copied = new ArrayList<RandomVariable>(vars);
		if (vars_copied.isEmpty()) {
			return 1;
		}
		RandomVariable Y = vars_copied.remove(0);
		if (e.variableSet().contains(Y)) {
			double prob = bn.getProbability(Y, e) * Ennumerateall(vars_copied, e, bn);
			return prob;
		} else {
			double summation = 0;
			Range y_range = Y.getRange();
			Iterator itr_y = y_range.iterator();
			while (itr_y.hasNext()) {
				Value value = (Value) itr_y.next();
				e.put(Y, value);
				summation += bn.getProbability(Y, e) * Ennumerateall(vars_copied, e, bn);
				e.remove(Y);
			}
			return summation;
		}
	}
}
