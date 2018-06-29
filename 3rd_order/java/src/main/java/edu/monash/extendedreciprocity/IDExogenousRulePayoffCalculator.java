package edu.monash.extendedreciprocity;

import com.evolutionandgames.jevodyn.PayoffCalculator;
import com.evolutionandgames.jevodyn.SimplePopulation;
import com.evolutionandgames.jevodyn.utils.Random;

public class IDExogenousRulePayoffCalculator implements PayoffCalculator {

	public static int NUMBER_OF_STRATEGIES = 4;
	public static int DEFECTOR_INDEX = 0;

	private UpdateRule updateRule;
	private double chi;
	private double epsilon;
	private double alpha;
	private double a;
	private double b;
	private double c;
	private double d;
	private int institutionNumber;
	private int numberOfRounds;

	private int[] institutionCode;
	private double[][] game = new double[2][2];

	enum UpdateRule {
		NO, OI
	}

	enum Strategies {
		Reciprocator, Cooperator, Defector, Contrarian
	}

	//private Strategies[] strategyList = { Strategies.Reciprocator, Strategies.Cooperator, Strategies.Defector,
	//		Strategies.Contrarian };
	
	private Strategies[] strategyList = {  Strategies.Defector, 
											Strategies.Cooperator,
											Strategies.Reciprocator,
											Strategies.Contrarian};
	
	
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

	public IDExogenousRulePayoffCalculator(UpdateRule updateRule, double chi, double epsilon, double alpha, double a,
			double b, double c, double d, int institutionNumber, int numberOfRounds) {

		this.updateRule = updateRule;
		this.chi = chi;
		this.epsilon = epsilon;
		this.alpha = alpha;
		this.a = a;
		this.b = b;
		this.c = c;
		this.d = d;
		this.institutionNumber = institutionNumber;
		this.numberOfRounds = numberOfRounds;
		
		this.game[0][0] = this.a;
		this.game[0][1] = this.b;
		this.game[1][0] = this.c;
		this.game[1][1] = this.d;
		
		this.institutionCode = IDExogenousRulePayoffCalculator.toBinary(this.institutionNumber);
	}

	public double[] getPayoff(SimplePopulation population) {
		int[] strategies = unpack(population);
		int[] reputations = new int[strategies.length];
		double[] fitness = new double[strategies.length];
		KahanSummation[] summations = new KahanSummation[strategies.length];
		for (int i = 0; i < summations.length; i++) {
			summations[i] = new KahanSummation();
		}
		
		
		for (int round = 1; round <= this.numberOfRounds; round++) {
			int[] encounters = Random.randomizeIndices(population.getSize());
			for (int i = 0; i < encounters.length-1; i=i+2) {
				
				
				int focalIndex = encounters[i];
				int otherIndex = encounters[i+1];
				
				//focal is index i
				Strategies focalStrategy = strategyList[strategies[focalIndex]];
				//other is index i+1
				Strategies otherStrategy = strategyList[strategies[otherIndex]];
				
				int focalReputation = reputations[focalIndex];
				int otherReputation = reputations[otherIndex];
				
				int focalReputationPerceivedByOther = focalReputation;
				int focalReputationPerceivedByThirdParty = focalReputation;
				int otherReputationPerceivedByFocal = otherReputation;
				int otherReputationPerceivedByThirdParty = otherReputation;
				
				
				//apply errors in judging reputations
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
				
				
				//figure out actions
				int focalAction = resolveAction(focalStrategy, otherReputationPerceivedByFocal);
				int otherAction = resolveAction(otherStrategy, focalReputationPerceivedByOther);
				
				//apply errors to actions
				if(focalAction == 0 && Random.bernoulliTrial(this.epsilon)){
					focalAction = 1 - focalAction;
				}
				if(focalAction == 0 && Random.bernoulliTrial(this.epsilon)){
					otherAction = 1 - otherAction;
				}
				
				
				
				//update payoffs
			
				//fitness[focalIndex] = fitness[focalIndex] + this.game[focalAction][otherAction];
				//fitness[otherIndex] = fitness[otherIndex] + this.game[otherAction][focalAction];
				summations[focalIndex].add(this.game[focalAction][otherAction]);
				summations[otherIndex].add(this.game[otherAction][focalAction]);
				
				
				//update reputations
				if (this.updateRule == UpdateRule.NO){
					reputations[focalIndex] = this.institutionCode[resolveIndexNO(focalAction, otherAction, otherReputationPerceivedByThirdParty)];
					//reputations[otherIndex] = this.institutionCode[resolveIndexNO(otherAction, focalAction, focalReputationPerceivedByThirdParty)];
				}
				if (this.updateRule == UpdateRule.OI){
					reputations[focalIndex] = this.institutionCode[resolveIndexOI(focalAction, focalReputationPerceivedByThirdParty, otherReputationPerceivedByThirdParty)];
					//reputations[otherIndex] = this.institutionCode[resolveIndexOI(otherAction, otherReputationPerceivedByThirdParty, focalReputationPerceivedByThirdParty)];
				}
				
				//apply errors to reputation assignment
				if (Random.bernoulliTrial(this.alpha)) {
					reputations[focalIndex] = 1- reputations[focalIndex];
				}
				//if (Random.bernoulliTrial(this.alpha)) {
			    // reputations[otherIndex] = 1- reputations[otherIndex];
		        //}
				
			}			
		}
		
		//fitness is an average over the number of rounds
		for (int i = 0; i < fitness.length; i++) {
			fitness[i] = summations[i].value() /numberOfRounds;
		}
		
		double[] payoff = repack(strategies, fitness, population.getNumberOfTypes());
		return payoff;
	}


	private int resolveIndexOI(int focalAction, int focalReputation, int otherReputation) {
		if(focalAction == 0 && focalReputation == 1 && otherReputation == 1){ return  0;}
		if(focalAction == 1 && focalReputation ==  1 && otherReputation ==  1){ return  1;}
		if(focalAction == 0 && focalReputation ==  1 && otherReputation ==  0){ return  2;}
		if(focalAction == 1 && focalReputation ==  1 && otherReputation ==  0){ return  3;}
		if(focalAction == 0 && focalReputation ==  0 && otherReputation ==  1){ return  4;}
		if(focalAction == 1 && focalReputation ==  0 && otherReputation ==  1){ return  5;}
		if(focalAction == 0 && focalReputation ==  0 && otherReputation ==  0){ return  6;}
		if(focalAction == 1 && focalReputation ==  0 && otherReputation ==  0){ return  7;}
		return -1;
	}

	
	private int resolveIndexNO(int focalAction, int otherAction, int otherReputation) {
		if(focalAction == 0 && otherAction == 0 && otherReputation == 1){ return  0;}
		if(focalAction == 0 && otherAction ==  1 && otherReputation ==  1){ return  1;}
		if(focalAction == 1 && otherAction ==  0 && otherReputation ==  1){ return  2;}
		if(focalAction == 1 && otherAction ==  1 && otherReputation ==  1){ return  3;}
		if(focalAction == 0 && otherAction ==  0 && otherReputation ==  0){ return  4;}
		if(focalAction == 0 && otherAction ==  1 && otherReputation ==  0){ return  5;}
		if(focalAction == 1 && otherAction ==  0 && otherReputation ==  0){ return  6;}
		if(focalAction == 1 && otherAction ==  1 && otherReputation ==  0){ return  7;}
		return -1;
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

	protected static double[] repack(int[] strategies, double[] fitness, int numberOfStrategies) {
		double[] ans = new double[numberOfStrategies];
		for (int i = 0; i < ans.length; i++) {
			double sumI = 0;
			int numberOfI = 0;
			for (int j = 0; j < strategies.length; j++) {
				if (strategies[j] == i){
					sumI = sumI + fitness[j];
					numberOfI++;
				}
			}
			if (numberOfI == 0) {
				ans[i] = 0;
			}else{
				ans[i] = sumI/numberOfI;
			}
		}
		return ans;
	}

	protected static int[] unpack(SimplePopulation population) {
		int[] array = population.getAsArrayOfTypes();
		int[] ans = new int[population.getSize()];
		int k = 0;		
		for (int i = 0; i < array.length; i++) {
			for (int j = 0; j < array[i]; j++) {
				ans[k] = i;
				k++;
			}
			
		}
		return ans;
	}

}
