//Developed by Hesham Elshafey and Ahmed Nassar
package bn.inference;

import java.util.Iterator;
import java.util.List;
import java.util.Random;

import bn.core.ApproximateInferencer;
import bn.core.Assignment;
import bn.core.BayesianNetwork;
import bn.base.Distribution;
import bn.core.Range;
import bn.core.Inferencer;
import bn.core.RandomVariable;
import bn.core.Value;

public class RejectionSampler implements ApproximateInferencer {

	public Distribution query(RandomVariable X, Assignment e, BayesianNetwork network, int n) {
		return Rejectionsampling(X, e, network, n);
	}

	static Assignment priorsample(BayesianNetwork bn, Random rng) {

		Assignment a = new bn.base.Assignment();
		List<RandomVariable> vars = bn.getVariablesSortedTopologically();

		for (RandomVariable V : vars) {
			Range v_range = V.getRange();
			Iterator itr_v = v_range.iterator();
			double rand_unif = rng.nextDouble();
			double cumprob = 0;
			
			while (itr_v.hasNext()) {
				Value value = (Value) itr_v.next();
				Assignment temp = a.copy();
				temp.put(V, value);
				double theprob = bn.getProbability(V, temp);
				double start = cumprob;
				double end = cumprob + theprob;
				cumprob = end;
				if (rand_unif >= start & rand_unif < end) {
					a.put(V, value);
				}
			}
		}
		return a.copy();

	}

	static Distribution Rejectionsampling(RandomVariable X, Assignment e, BayesianNetwork bn, int nsamples) {
		Random rng = new Random();
		Distribution dist = new Distribution(X);
		Range X_range = X.getRange();
		Iterator<Value> itr_X = X_range.iterator();

		while (itr_X.hasNext()) {
			Value v = (Value) itr_X.next();
			dist.set(v, 1);
		}

		for (int i = 0; i < nsamples; i++) {
			Assignment x = priorsample(bn, rng);
			if (x.containsAll(e)) {
				Value v = x.get(X);
				double previous_score = dist.get(v);
				double new_score = 1 + previous_score;
				dist.remove(v);
				dist.set(v, new_score);
			}
		}
		return dist;
	}

}
