package edu.monash.extendedreciprocity.imitation;

import java.util.Arrays;

import com.evolutionandgames.jevodyn.EvolutionaryProcess;
import com.evolutionandgames.jevodyn.SimplePopulation;
import com.evolutionandgames.jevodyn.impl.SimplePopulationImpl;
import com.evolutionandgames.jevodyn.utils.Random;



public class ImitationProcess implements EvolutionaryProcess{
	
	
	
	
	
	private UpdateRule updateRule;
	private double chi;
	private double epsilon;
	private double alpha;
	private double a;
	private double b;
	private double c;
	private double d;
	private int institutionNumber;
	private double[][] game = new double[2][2];
	private ImitationPopulation population;
	private double mutationProbability;
	private int populationSize;
	private double intensityOfSelection;	
	private int timeStep;

	
	enum UpdateRule {
		NO, OI
	}

	
	public ImitationProcess(UpdateRule updateRule, double chi, double epsilon, double alpha, double a,
			double b, double c, double d, int institutionNumber, int populationSize, double mutationProbability, double intensityOfSelection) {
		super();
		this.updateRule = updateRule;
		this.chi = chi;
		this.epsilon = epsilon;
		this.alpha = alpha;
		this.a = a;
		this.b = b;
		this.c = c;
		this.d = d;
		this.institutionNumber = institutionNumber;
		
		this.game[0][0] = this.a;
		this.game[0][1] = this.b;
		this.game[1][0] = this.c;
		this.game[1][1] = this.d;
		this.populationSize = populationSize;
		this.intensityOfSelection = intensityOfSelection;
		this.population = new ImitationPopulation(populationSize, this.chi, this.epsilon, this.alpha, this.a, this.b, this.c, this.d, this.updateRule, this.institutionNumber, this.intensityOfSelection);
		this.mutationProbability = mutationProbability;
		this.timeStep = 0;
	}

	

	
	
	
	
	










	public void step(){
		int indexA = Random.nextInt(population.getSize());
		if (Random.bernoulliTrial(this.mutationProbability)) {
			// set individual A to random strategy
			this.population.setToRandomStrategy(indexA);	
		}else{
			
			// Pick random B that is not A
			int indexB = ImitationProcess.pickExcept(indexA, population.getSize());

			// set fitness to zero for everybody
			population.resetFitness();

			// Following SSP I fix the number of rounds as twice pop-size
			for (int i = 0; i < 2*population.getSize(); i++) {
				
				// Evaluate A fitness
				int indexC = pickExcept(indexA, population.getSize());
				// play round will update payoff and reputation of both a and c
				population.playRound(indexA, indexC);
			
				// Evaluate B fitness
				indexC = pickExcept(indexB, population.getSize());
				// play round will update payoff and reputation of both a and c
				population.playRound(indexB, indexC);
				
			}
			population.setFitness(indexA, population.getFitness(indexA)/(2.0*this.population.getSize()));
			population.setFitness(indexB, population.getFitness(indexB)/(2.0*this.population.getSize()));
			population.imitate(indexA, indexB);
			this.timeStep++;
			
			
		}
		
		
		
		
	
		
	}

	/**
	 * Picks an integer in interval [0, maximumExclusive) with uniform probability excluding index 
	 * @param index
	 * @param maximumExclusive
	 * @return
	 */
	public static int pickExcept(int index, int maximumExclusive) {
		double[] distribution = new double[maximumExclusive];
		Arrays.fill(distribution, 1.0/(maximumExclusive-1.0));
		distribution[index] = 0.0;
		return Random.simulateDiscreteDistribution(distribution);
	}





	public void stepWithoutMutation() {
		System.out.println("Should never be called");
		System.exit(0);
	}



	public SimplePopulation getPopulation() {
		int[] populationArray = this.population.getCounts();
		return new SimplePopulationImpl(populationArray);
	}


	public void reset(SimplePopulation startingPopulation) {
		this.timeStep = 0;
		this.population = new ImitationPopulation(populationSize, this.chi, this.epsilon, this.alpha, this.a, this.b, this.c, this.d, this.updateRule, this.institutionNumber, this.intensityOfSelection);		
	}









	public double getTotalPopulationPayoff() {
		return this.population.averagePayoff();
	}









	public int getTimeStep() {
		return this.timeStep;
	}


	public void setKeepTrackTotalPayoff(boolean keepTrack) {
		//Nothing to do
		
	}
	
		
		
	

}
