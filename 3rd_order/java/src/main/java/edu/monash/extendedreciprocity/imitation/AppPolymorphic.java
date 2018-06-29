package edu.monash.extendedreciprocity.imitation;

import java.io.File;
import java.io.IOException;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.evolutionandgames.jevodyn.Simulation;
import com.evolutionandgames.jevodyn.utils.Random;
import com.google.common.base.Charsets;
import com.google.common.base.Joiner;
import com.google.common.io.Files;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import edu.monash.extendedreciprocity.imitation.ImitationProcess.UpdateRule;

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

	private Long seed = null;
	private String outputFileName;

	private int reportEveryTimeSteps;
	private int numberOfTimeSteps;
	private int numberOfRuns;

	public static String exampleJsonTime() {
		String json = new GsonBuilder().setPrettyPrinting().create().toJson(exampleAppTime());
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
		app.a = 4;
		app.b = -1;
		app.c = 5;
		app.d = 0;
		app.institutionNumber = 15;
		// app.numberOfRounds=100;
		app.outputFileName = "example.csv";
		app.numberOfTimeSteps = 20000;
		app.reportEveryTimeSteps = 50;
		app.seed = (long) 6666;
		app.numberOfRuns = 100;
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
			AppPolymorphic.runAppTimeSeries(application);
		}

	}

	protected static void runAppTimeSeries(AppPolymorphic app) throws IOException {
		if (app.seed == null) {
			app.seed = System.currentTimeMillis();
		}
		Random.seed(app.seed);
		try {
			String corename = app.outputFileName.substring(0, app.outputFileName.lastIndexOf('.'));
			String extension = app.outputFileName.substring(app.outputFileName.lastIndexOf('.') + 1,
					app.outputFileName.length());

			for (int run = 0; run < app.numberOfRuns; run++) {

				ImitationProcess process = new ImitationProcess(app.updateRule, app.chi, app.epsilon, app.alpha, app.a,
						app.b, app.c, app.d, app.institutionNumber, app.populationSize, app.mutationProbability,
						app.intensityOfSelection);

				Simulation sim = new Simulation(process);
				sim.simulateTimeSeries(app.numberOfTimeSteps, app.reportEveryTimeSteps, app.seed, true,
						corename + "_run_" + run + "." + extension);
			}

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

	private static AppPolymorphic loadFromFile(String jsonFile) throws IOException {
		File file = new File(jsonFile);
		Gson gson = new Gson();
		String json = Files.toString(file, Charsets.UTF_8);
		AppPolymorphic app = gson.fromJson(json, AppPolymorphic.class);
		return app;
	}

}
