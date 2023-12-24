package bn.inference;

import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

import bn.base.Distribution;
import bn.base.StringValue;
import bn.core.ApproximateInferencer;
import bn.core.Assignment;
import bn.core.BayesianNetwork;
import bn.core.Inferencer;
import bn.core.RandomVariable;
import bn.core.Value;
import bn.inference.EnumerationInferencer;
import bn.parser.XMLBIFParser;

public class Main {
	public static void main(String[] argv) throws IOException, ParserConfigurationException, SAXException {
		if (argv[0].equals("MyBNInferencer")) {
			String filename = argv[1];
			XMLBIFParser parser = new XMLBIFParser();
			BayesianNetwork network = parser.readNetworkFromFile(filename);
			
			System.out.println(network);
			
			Assignment e = new bn.base.Assignment();
			RandomVariable Q = network.getVariableByName(argv[2]);

			for (int i = 3; i < argv.length; i += 2) {
				RandomVariable E = network.getVariableByName(argv[i]);
				String str = argv[i + 1];
				if (str.equals("true") || str.equals("false")) {
					Value val = new StringValue(str);
					e.put(E, val);
				} else {
					Value v = new StringValue(str);
					e.put(E, v);
				}
			}

			Inferencer exact = new EnumerationInferencer();
			Distribution dist = (Distribution) exact.query(Q, e.copy(), network);
			dist.normalize();
			System.out.println(dist);
		}

		else {
			
			String filename = argv[2];
			XMLBIFParser parser = new XMLBIFParser();
			BayesianNetwork network = parser.readNetworkFromFile(filename);
			
			
			int n = Integer.parseInt(argv[1]);
			Assignment e = new bn.base.Assignment();
			RandomVariable Q = network.getVariableByName(argv[3]);

			System.out.println(network);
			
			for (int i = 4; i < argv.length; i += 2) {
				RandomVariable E = network.getVariableByName(argv[i]);
				String str = argv[i + 1];
				if (str.equals("true") || str.equals("false")) {
					Value val = new StringValue(str);
					e.put(E, val);
				} else {
					Value v = new StringValue(str);
					e.put(E, v);
				}
			}

			ApproximateInferencer approx = new RejectionSampler();
			Distribution dist1 = (Distribution) approx.query(Q, e.copy(), network, n);
			dist1.normalize();

			approx = new LikelihoodSampler();
			Distribution dist2 = (Distribution) approx.query(Q, e.copy(), network, n);
			dist2.normalize();

			System.out.println("Rejection Sampling");
			System.out.println(dist1);
			System.out.println("Likelihood Sampling:");
			System.out.println(dist2);
		}

	}
}