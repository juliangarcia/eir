package edu.monash.extendedreciprocity.imitation;

import java.io.FileWriter;
import java.io.IOException;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.List;

import org.supercsv.cellprocessor.constraint.NotNull;
import org.supercsv.cellprocessor.ift.CellProcessor;
import org.supercsv.io.CsvListWriter;
import org.supercsv.io.ICsvListWriter;
import org.supercsv.prefs.CsvPreference;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.evolutionandgames.jevodyn.utils.Random;

import edu.monash.extendedreciprocity.imitation.ImitationProcess.UpdateRule;

public class ImitationDynamicsInspector {

	// PARAMETERS
	@Parameter(names = { "--pop_size", "-Z" }, description = "Name of the json file", required = true)
	private int populationSize; // = 70;

	@Parameter(names = { "--chi" }, description = "chi")
	private double chi = 0.0;
	@Parameter(names = { "--epsilon" }, description = "epsilon")
	private double epsilon = 0.08;
	@Parameter(names = { "--alpha" }, description = "alpha")
	private double alpha = 0.01;
	@Parameter(names = { "--R" }, description = "R")
	private double R = 4.0;
	@Parameter(names = { "--S" }, description = "S")
	private double S = -1.0;
	@Parameter(names = { "--T" }, description = "T")
	private double T = 5.0;
	@Parameter(names = { "--P" }, description = "P")
	private double P = 0.0;
	@Parameter(names = { "--institution_code" }, description = "Institution", required = true)
	private int institutionNumber; // = 9;
	@Parameter(names = { "--ios" }, description = "Intensity")
	private double intensityOfSelection = 1.0;
	@Parameter(names = { "--mutant_index" }, description = "Type A", required = true)
	private int typeA; // = 0;
	@Parameter(names = { "--resident_index" }, description = "Type B", required = true)
	private int typeB; // = 2;
	@Parameter(names = { "--number_of_mutants" }, description = "number of Type A", required = true)
	private int numberTypeA; // = 2; // 2 defectors, 70-2 reciprocators

	public String toString() {
		return "inst_" + this.institutionNumber + "_chi_" + this.chi + "_epsilon_" + this.epsilon + "_alpha_"
				+ this.alpha + "_chi_" + this.chi + "_R_" + this.R + "_S_" + this.S + "_T_" + this.T + "_P_" + this.P
				+ "_ios_" + this.intensityOfSelection + "_type_A_" + this.typeA + "_type_B_" + this.typeB + "_n_A_"
				+ this.numberTypeA + "_Z_" +this.populationSize;
	}

	public static void main(String[] args) throws IOException {

		ImitationDynamicsInspector app = new ImitationDynamicsInspector();
		JCommander commander = new JCommander(app);
		try {
			commander.parse(args);
		} catch (ParameterException e) {
			System.out.println(e.getMessage());
			commander.usage();
			return;
		}

		UpdateRule updateRule = ImitationProcess.UpdateRule.OI;

		Random.seed();

		// Defector, Cooperator, Reciprocator,Contrarian
		// int typeA = 0;
		// int typeB = 2;
		// int numberTypeA = 2; // 2 defectors, 70-2 reciprocators
		String fileName = app.toString() + ".csv";

		if (app.numberTypeA > app.populationSize) {
			throw new InvalidParameterException("number of Type A < pop Size?");
		}
		int[] strategies = new int[app.populationSize];
		for (int i = 0; i < app.numberTypeA; i++) {
			strategies[i] = app.typeA;
		}
		for (int i = app.numberTypeA; i < app.populationSize; i++) {
			strategies[i] = app.typeB;
		}
		ImitationPopulation population = new ImitationPopulation(app.populationSize, app.chi, app.epsilon, app.alpha,
				app.R, app.S, app.T, app.P, updateRule, app.institutionNumber, app.intensityOfSelection);
		population.setStrategies(strategies);

		ICsvListWriter listWriter = null;
		String[] header = buildHeader(app.populationSize);
		CellProcessor[] processors = buildProcessors(app.populationSize);

		try {

			listWriter = new CsvListWriter(new FileWriter(fileName), CsvPreference.STANDARD_PREFERENCE);
			listWriter.writeHeader(header);
			int indexA = Random.nextInt(population.getSize());
			int indexB = ImitationProcess.pickExcept(indexA, app.populationSize);
			// set fitness to zero for everybody
			population.resetFitness();
			// Following SSP I fix the number of rounds as twice pop-size
			for (int i = 0; i < 2 * population.getSize(); i++) {

				// Evaluate A fitness
				int indexC = ImitationProcess.pickExcept(indexA, population.getSize());
				// play round will update payoff and reputation of both a and c
				population.playRound(indexA, indexC);

				// Evaluate B fitness
				indexC = ImitationProcess.pickExcept(indexB, population.getSize());
				// play round will update payoff and reputation of both a and c
				population.playRound(indexB, indexC);
				// Write things down
				listWriter.write(currentStateRow(population, i), processors);

			}
		} finally {
			// close files no matter what
			if (listWriter != null) {
				listWriter.close();
			}
		}

	}

	private static List<Object> currentStateRow(ImitationPopulation population, int timeStep) {
		ArrayList<Object> myList = new ArrayList<Object>();
		myList.add(timeStep);
		int[] reputations = population.getReputations();
		for (int i = 0; i < reputations.length; i++) {
			myList.add(reputations[i]);
		}
		int[] strategies = population.getStrategies();
		for (int i = 0; i < strategies.length; i++) {
			myList.add(strategies[i]);
		}

		return myList;
	}

	private static CellProcessor[] buildProcessors(int populationSize) {
		final CellProcessor[] processors = new CellProcessor[populationSize * 2 + 1];
		for (int i = 0; i < 2 * populationSize; i++) {
			processors[i] = new NotNull();
		}
		return processors;
	}

	private static String[] buildHeader(int populationSize) {
		ArrayList<String> myList = new ArrayList<String>();
		myList.add("timeStep");
		for (int i = 1; i <= populationSize; i++) {
			myList.add("R" + Integer.valueOf(i).toString());
		}
		for (int i = 1; i <= populationSize; i++) {
			myList.add("T" + Integer.valueOf(i).toString());
		}
		String[] ans = new String[myList.size()];
		for (int i = 0; i < ans.length; i++) {
			ans[i] = myList.get(i);
		}
		return ans;
	}

}
