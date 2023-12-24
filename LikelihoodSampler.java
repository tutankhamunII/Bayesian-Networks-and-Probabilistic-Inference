//Developed by Hesham Elshafey and Ahmed Nassar
package bn.inference;

import bn.core.ApproximateInferencer;
import bn.core.Assignment;
import bn.core.BayesianNetwork;
import bn.core.Range;

import java.util.Iterator;
import java.util.List;
import java.util.Random;

import bn.base.Distribution;
import bn.core.Inferencer;
import bn.core.RandomVariable;
import bn.core.Value;

public class LikelihoodSampler implements ApproximateInferencer {

	@Override
	public Distribution query(RandomVariable X, Assignment e, BayesianNetwork network, int n) {
		return likehoodweighting(X,e,network, n);
	}

	static Distribution likehoodweighting(RandomVariable X, Assignment e, BayesianNetwork bn, int nsamples) {
		Distribution dist = new Distribution(X);
		Range X_range = X.getRange();
		Iterator itr_X = X_range.iterator();
		//System.out.println(e.toString());

		while (itr_X.hasNext()) {
			Value v = (Value) itr_X.next();

			dist.set(v, 0);
		}

		for (int i = 0; i < nsamples; i++) {

			Object[] arr = weightedsample(bn, e);
			Assignment x = (bn.base.Assignment) arr[1];
			Double w = (Double) arr[0];

			Value v = x.get(X);
			double previous_score = dist.get(v);

			double new_score = w + previous_score;
			dist.remove(v);
			dist.set(v, new_score);
		}

		return dist;

	}

	static Object[] weightedsample(BayesianNetwork bn, Assignment obs) {
		Object[] arr = new Object[2];
		Double w = 1.0;
		// Assignment ec = e.copy();
		Assignment toreturn = obs.copy();
		List<RandomVariable> vars = bn.getVariablesSortedTopologically();

		for (RandomVariable V : vars) {

			if (obs.variableSet().contains(V)) {
				w = w * bn.getProbability(V, toreturn);
				toreturn.put(V, obs.get(V));

			} else {

				Range v_range = V.getRange();
				Iterator itr_v = v_range.iterator();

				Random rng = new Random();
				double rand_unif = rng.nextDouble();

				double cumprob = 0;

				while (itr_v.hasNext()) {
					Value value = (Value) itr_v.next();

					Assignment temp = toreturn.copy();

					temp.put(V, value);
					double theprob = bn.getProbability(V, temp);

					double start = cumprob;

					double end = cumprob + theprob;
					cumprob = end;

					if (rand_unif >= start & rand_unif < end) {

						toreturn.put(V, value);
					}

				}

			}
		}
		arr[0] = (Double) w;
		arr[1] = (Assignment) toreturn.copy();
		return arr;
	}
		

}
