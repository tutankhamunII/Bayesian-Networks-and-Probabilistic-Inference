package bn.core;

public interface ApproximateInferencer {
	public Distribution query(RandomVariable X, Assignment e, BayesianNetwork network, int n);
}
