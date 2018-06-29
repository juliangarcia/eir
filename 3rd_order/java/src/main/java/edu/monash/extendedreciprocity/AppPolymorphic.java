package edu.monash.extendedreciprocity;

import java.io.File;
import java.io.IOException;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.evolutionandgames.jevodyn.PayoffCalculator;
import com.evolutionandgames.jevodyn.SimplePopulation;
import com.evolutionandgames.jevodyn.Simulation;
import com.evolutionandgames.jevodyn.impl.MoranProcess;
import com.evolutionandgames.jevodyn.impl.SimplePopulationImpl;
import com.evolutionandgames.jevodyn.utils.PayoffToFitnessMapping;
import com.evolutionandgames.jevodyn.utils.Random;
import com.google.common.base.Charsets;
import com.google.common.base.Joiner;
import com.google.common.io.Files;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import edu.monash.extendedreciprocity.IDExogenousRulePayoffCalculator.UpdateRule;


public class AppPolymorphic {
	
	public static boolean RANDOM = true;
	
	
	// transient configuration members
	@Parameter(names = { "-file", "-f" }, description = "Name of the json file")
	private transient String file;

	@Parameter(names = "-json", description = "Show json")
	private transient boolean json = false;


	

	// /PARAMETERS
	private double mutationProbability;
	private double intensityOfSelection;
	private int populationSize;
	
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
	

	
	private Long seed = null;
	private String outputFileName;
	
	
	
	
	private int reportEveryTimeSteps;
	private int numberOfTimeSteps;

	

	

	

	public static String exampleJsonTime() {
		String json = new GsonBuilder().setPrettyPrinting().create()
				.toJson(exampleAppTime());
		return json;
	}

	public static AppPolymorphic exampleAppTime() {
		AppPolymorphic app = new AppPolymorphic();
		app.mutationProbability = 0.01;
		app.intensityOfSelection = 1.0;
		app.populationSize = 50;
		app.updateRule = UpdateRule.NO;
		app.chi = 0.01;
		app.epsilon = 0.08;
		app.alpha = 0.01;
		app.a=4;
		app.b=-1;
		app.c=5;
		app.d=0;
		app.institutionNumber=15;
		app.numberOfRounds=100;
		app.outputFileName = "example.csv";
		app.numberOfTimeSteps = 20000;
		app.reportEveryTimeSteps = 50;
		app.seed = (long) 6666;
		return app;
	}

	public static void main(String[] args) throws IOException {

		AppPolymorphic app = new AppPolymorphic();

		// Parsing
		JCommander commander = new JCommander(app);
		try {
			commander.parse(args);
		} catch (ParameterException e) {
			System.out.println(e.getMessage());
			commander.usage();
			return;
		}
		// Done with parsing, get to business.
		if (app.json) {
				System.out.println(AppPolymorphic.exampleJsonTime());
			} else {
				AppPolymorphic application = AppPolymorphic.loadFromFile(app.file);
				AppPolymorphic.runAppTimeSeries(application );
			}
		
	}

	
	protected static void runAppTimeSeries(AppPolymorphic app)
			throws IOException {
		if (app.seed == null) {
			app.seed = System.currentTimeMillis();
		}
		Random.seed(app.seed);

		PayoffCalculator payoffCalculator = new IDExogenousRulePayoffCalculator(app.updateRule, app.chi, app.epsilon, 
				app.alpha, app.a, app.b, app.c, app.d, app.institutionNumber, app.numberOfRounds);

		int[] populationArray = new int[IDExogenousRulePayoffCalculator.NUMBER_OF_STRATEGIES];
		
		if (AppPolymorphic.RANDOM){
			//Randomly allocate initial population
			for (int i = 0; i < app.populationSize; i++) {
				int index = Random.nextInt(IDExogenousRulePayoffCalculator.NUMBER_OF_STRATEGIES);
				populationArray[index] = populationArray[index] +1; 
			}
		}else{
			populationArray[IDExogenousRulePayoffCalculator.DEFECTOR_INDEX] = app.populationSize;
		}
		
		try {
			
		SimplePopulation population = new SimplePopulationImpl(populationArray);
		MoranProcess mp = new MoranProcess(population, payoffCalculator,
				PayoffToFitnessMapping.EXPONENTIAL, app.mutationProbability,
				app.intensityOfSelection);
		Simulation sim = new Simulation(mp);
		sim.simulateTimeSeries(app.numberOfTimeSteps, app.reportEveryTimeSteps,
				app.seed, true, app.outputFileName);
		} catch (RuntimeException e) {
			System.out.println("Error producing: " + app.outputFileName);
			e.printStackTrace();
		}

	}

	protected static String buildString(double[] ans) {
		Joiner joiner = Joiner.on(",").useForNull("0.0");
		Double[] doubleArray = new Double[ans.length];
		for (int i = 0; i < ans.length; i++) {
			doubleArray[i] = ans[i];
		}
		return joiner.join(doubleArray);
	}

	private static AppPolymorphic loadFromFile(String jsonFile)
			throws IOException {
		File file = new File(jsonFile);
		Gson gson = new Gson();
		String json = Files.toString(file, Charsets.UTF_8);
		AppPolymorphic app = gson.fromJson(json, AppPolymorphic.class);
		return app;
	}

	
	

}
