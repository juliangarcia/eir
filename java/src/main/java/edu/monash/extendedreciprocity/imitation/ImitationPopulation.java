package edu.monash.extendedreciprocity.imitation;

import java.util.Arrays;

import com.evolutionandgames.jevodyn.utils.Random;

import edu.monash.extendedreciprocity.imitation.ImitationProcess.UpdateRule;

public class ImitationPopulation {

	private int[] strategies;
	private int[] reputations;
	private double[] fitness;
	private double chi;
	private double epsilon;
	private double alpha;
	public int[] getStrategies() {
		return strategies;
	}

	public int[] getReputations() {
		return reputations;
	}

	public double[] getFitness() {
		return fitness;
	}

	private double[][] game = new double[2][2];
	private UpdateRule updateRule;
	private int institutionNumber;
	private int[] institutionCode;
	private long numberR = 0;

	/**
	 * @return the numberR
	 */
	public long getNumberR() {
		return numberR;
	}

	/**
	 * @return the numberS
	 */
	public long getNumberS() {
		return numberS;
	}

	/**
	 * @return the numberT
	 */
	public long getNumberT() {
		return numberT;
	}

	/**
	 * @return the numberP
	 */
	public long getNumberP() {
		return numberP;
	}

	public double averagePayoff() {
		try {
			double total = this.numberR + this.numberS + this.numberT + this.numberP;
			double freqA = ((double) this.numberR / total);
			double freqB = ((double) this.numberS / total);
			double freqC = ((double) this.numberT / total);
			double freqD = ((double) this.numberP / total);
			return freqA * this.game[0][0] + freqB * this.game[0][1] + freqC * this.game[1][0]
					+ freqD * this.game[1][1];
		} catch (ArithmeticException e) {
			return 0;
		}
	}

	private long numberS = 0;
	private long numberT = 0;
	private long numberP = 0;
	private double intensityOfSelection;

	public ImitationPopulation(int populationSize, double chi, double epsilon, double alpha, double a, double b,
			double c, double d, UpdateRule updateRule, int institutionNumber, double intensityOfSelection) {
		super();
		this.strategies = new int[populationSize];
		this.reputations = new int[populationSize];
		this.fitness = new double[populationSize];
		// initialize strategies and reputations randomly
		for (int i = 0; i < strategies.length; i++) {
			strategies[i] = Random.nextInt(NUMBER_OF_STRATEGIES);
			reputations[i] = Random.nextInt(2);
		}
		this.chi = chi;
		this.epsilon = epsilon;

		this.game[0][0] = a;
		this.game[0][1] = b;
		this.game[1][0] = c;
		this.game[1][1] = d;
		this.updateRule = updateRule;
		this.institutionNumber = institutionNumber;
		this.institutionCode = ImitationPopulation.toBinary(this.institutionNumber);
		this.alpha = alpha;
		this.intensityOfSelection = intensityOfSelection;

	}

	public void setStrategies(int[] strategies) {
		this.strategies = strategies;
	}

	public static int NUMBER_OF_STRATEGIES = 4;
	public static int DEFECTOR_INDEX = 0;

	public static Strategies[] strategyList = { Strategies.Defector, Strategies.Cooperator, Strategies.Reciprocator,
			Strategies.Contrarian };

	enum Strategies {
		Reciprocator, Cooperator, Defector, Contrarian
	}

	public int getSize() {
		return this.strategies.length;
	}

	public void setToRandomStrategy(int index) {
		this.strategies[index] = Random.nextInt(NUMBER_OF_STRATEGIES);
	}

	public double getFitness(int index) {
		return this.fitness[index];
	}

	public void setFitness(int index, double value) {
		this.fitness[index] = value;
	}

	public void resetFitness() {
		Arrays.fill(this.fitness, 0.0);
	}

	public void playRound(int focalIndex, int otherIndex) {

		// focal is index i
		Strategies focalStrategy = strategyList[strategies[focalIndex]];
		// other is index i+1
		Strategies otherStrategy = strategyList[strategies[otherIndex]];

		int focalReputation = this.reputations[focalIndex];
		int otherReputation = this.reputations[otherIndex];

		int focalReputationPerceivedByOther = focalReputation;
		int focalReputationPerceivedByThirdParty = focalReputation;
		int otherReputationPerceivedByFocal = otherReputation;
		int otherReputationPerceivedByThirdParty = otherReputation;

		// apply errors in judging reputations
		if (Random.bernoulliTrial(this.chi)) {
			focalReputationPerceivedByOther = 1 - focalReputationPerceivedByOther;
		}
		if (Random.bernoulliTrial(this.chi)) {
			focalReputationPerceivedByThirdParty = 1 - focalReputationPerceivedByThirdParty;
		}
		if (Random.bernoulliTrial(this.chi)) {
			otherReputationPerceivedByFocal = 1 - otherReputationPerceivedByFocal;
		}
		if (Random.bernoulliTrial(this.chi)) {
			otherReputationPerceivedByThirdParty = 1 - otherReputationPerceivedByThirdParty;
		}

		// figure out actions
		int focalAction = resolveAction(focalStrategy, otherReputationPerceivedByFocal);
		int otherAction = resolveAction(otherStrategy, focalReputationPerceivedByOther);

		// apply errors to actions
		if (focalAction == 0 && Random.bernoulliTrial(this.epsilon)) {
			focalAction = 1 - focalAction;
		}
		if (focalAction == 0 && Random.bernoulliTrial(this.epsilon)) {
			otherAction = 1 - otherAction;
		}

		// update payoffs by adding (no longer Kahan)
		fitness[focalIndex] = fitness[focalIndex] + this.game[focalAction][otherAction];

		// no need to update c fitness
		// fitness[otherIndex] = fitness[otherIndex] +
		// this.game[otherAction][focalAction];

		if (focalAction == 0 && otherAction == 0) {
			this.numberR++;
		}
		if (focalAction == 0 && otherAction == 1) {
			this.numberS++;
		}
		if (focalAction == 1 && otherAction == 0) {
			this.numberT++;
		}
		if (focalAction == 1 && otherAction == 1) {
			this.numberP++;
		}

		// update reputations
		if (this.updateRule == UpdateRule.NO) {
			reputations[focalIndex] = this.institutionCode[resolveIndexNO(focalAction, otherAction,
					otherReputationPerceivedByThirdParty)];
			// reputations[otherIndex] =
			// this.institutionCode[resolveIndexNO(otherAction, focalAction,
			// focalReputationPerceivedByThirdParty)];
		}
		if (this.updateRule == UpdateRule.OI) {
			reputations[focalIndex] = this.institutionCode[resolveIndexOI(focalAction,
					focalReputationPerceivedByThirdParty, otherReputationPerceivedByThirdParty)];
			//TODO: This below was commented.
			reputations[otherIndex] = this.institutionCode[resolveIndexOI(otherAction,
					otherReputationPerceivedByThirdParty, focalReputationPerceivedByThirdParty)];
		}

		// apply errors to reputation assignment
		if (Random.bernoulliTrial(this.alpha)) {
			reputations[focalIndex] = 1 - reputations[focalIndex];
		}

	}

	private int resolveAction(Strategies focalStrategy, int reputation) {
		switch (focalStrategy) {
		case Reciprocator:
			if (reputation == 0) {
				return 1;
			}
			if (reputation == 1) {
				return 0;
			}

		case Cooperator:
			if (reputation == 0) {
				return 0;
			}
			if (reputation == 1) {
				return 0;
			}
		case Defector:
			if (reputation == 0) {
				return 1;
			}
			if (reputation == 1) {
				return 1;
			}
		case Contrarian:
			if (reputation == 0) {
				return 0;
			}
			if (reputation == 1) {
				return 1;
			}
		}
		// should never get here.
		return -1;
	}

	protected static int[] toBinary(int institutionNumber) {
		int[] ans = new int[8];

		String binaryString = Integer.toBinaryString(institutionNumber);
		int offSet = 8 - binaryString.length();
		int i = 0;
		for (int j = offSet; j < ans.length; j++) {
			ans[j] = (binaryString.charAt(i) == '1') ? 1 : 0;
			i++;
		}
		return ans;
	}

	private int resolveIndexOI(int focalAction, int focalReputation, int otherReputation) {
		if (focalAction == 0 && focalReputation == 1 && otherReputation == 1) {
			return 0;
		}
		if (focalAction == 1 && focalReputation == 1 && otherReputation == 1) {
			return 1;
		}
		if (focalAction == 0 && focalReputation == 1 && otherReputation == 0) {
			return 2;
		}
		if (focalAction == 1 && focalReputation == 1 && otherReputation == 0) {
			return 3;
		}
		if (focalAction == 0 && focalReputation == 0 && otherReputation == 1) {
			return 4;
		}
		if (focalAction == 1 && focalReputation == 0 && otherReputation == 1) {
			return 5;
		}
		if (focalAction == 0 && focalReputation == 0 && otherReputation == 0) {
			return 6;
		}
		if (focalAction == 1 && focalReputation == 0 && otherReputation == 0) {
			return 7;
		}
		return -1;
	}

	private int resolveIndexNO(int focalAction, int otherAction, int otherReputation) {
		if (focalAction == 0 && otherAction == 0 && otherReputation == 1) {
			return 0;
		}
		if (focalAction == 0 && otherAction == 1 && otherReputation == 1) {
			return 1;
		}
		if (focalAction == 1 && otherAction == 0 && otherReputation == 1) {
			return 2;
		}
		if (focalAction == 1 && otherAction == 1 && otherReputation == 1) {
			return 3;
		}
		if (focalAction == 0 && otherAction == 0 && otherReputation == 0) {
			return 4;
		}
		if (focalAction == 0 && otherAction == 1 && otherReputation == 0) {
			return 5;
		}
		if (focalAction == 1 && otherAction == 0 && otherReputation == 0) {
			return 6;
		}
		if (focalAction == 1 && otherAction == 1 && otherReputation == 0) {
			return 7;
		}
		return -1;
	}

	public int[] getCounts() {
		// def coop recip paradox
		int countD = 0;
		int countC = 0;
		int countR = 0;
		int countP = 0;
		for (int i = 0; i < strategies.length; i++) {
			switch (strategyList[strategies[i]]) {
			case Defector:
				countD++;
				break;
			case Cooperator:
				countC++;
				break;
			case Reciprocator:
				countR++;
				break;
			case Contrarian:
				countP++;
				break;
			default:
				return null;
			}
		}
		int[] ans = { countD, countC, countR, countP };
		return ans;
	}

	public void imitate(int imitatorIndex, int modelIndex) {
		double payoffImitator = this.fitness[imitatorIndex];
		double payoffModel = this.fitness[modelIndex];
		double p = 1.0 / (1 + Math.exp(this.intensityOfSelection * (payoffImitator - payoffModel)));
		if (Random.bernoulliTrial(p)) {
			this.strategies[imitatorIndex] = this.strategies[modelIndex];
		}
	}

}
