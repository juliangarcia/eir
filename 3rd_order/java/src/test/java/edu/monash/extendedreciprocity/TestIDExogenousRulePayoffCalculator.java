package edu.monash.extendedreciprocity;

import org.junit.Assert;
import org.junit.Test;

import com.evolutionandgames.jevodyn.impl.SimplePopulationImpl;
import com.evolutionandgames.jevodyn.utils.Random;

import edu.monash.extendedreciprocity.IDExogenousRulePayoffCalculator.UpdateRule;

public class TestIDExogenousRulePayoffCalculator {

	//@Test
	public void testBinaryCode() {
		//IDExogenousRulePayoffCalculator payoffCalculator = new IDExogenousRulePayoffCalculator(updateRule, chi, epsilon, alpha, a, b, c, d, institutionNumber, numberOfRounds)
		int[] actuals = IDExogenousRulePayoffCalculator.toBinary(8);
		int[] expecteds = {0, 0, 0, 0, 1, 0, 0, 0};
		Assert.assertArrayEquals(expecteds, actuals);
		
		actuals = IDExogenousRulePayoffCalculator.toBinary(253);
		int[] expecteds2 = {1, 1, 1, 1, 1, 1, 0, 1};
		Assert.assertArrayEquals(expecteds2, actuals);
		
		
	}
	
	//@Test
	public void testUnpack() {
		//IDExogenousRulePayoffCalculator payoffCalculator = new IDExogenousRulePayoffCalculator(updateRule, chi, epsilon, alpha, a, b, c, d, institutionNumber, numberOfRounds)
		
		int[] populationArray = {2, 3, 1};
		SimplePopulationImpl population = new SimplePopulationImpl(populationArray);
		int[] actuals = IDExogenousRulePayoffCalculator.unpack(population);
		int[] expecteds = {0, 0, 1, 1, 1, 2};
		Assert.assertArrayEquals(expecteds, actuals);
		
		
		int[] populationArray2 = {0, 0, 1, 1};
		SimplePopulationImpl population2 = new SimplePopulationImpl(populationArray2);
		int[] actuals2 = IDExogenousRulePayoffCalculator.unpack(population2);
		int[] expecteds2 = {2, 3};
		Assert.assertArrayEquals(expecteds2, actuals2);
		
		
		int[] populationArray3 = {5, 0, 0, 0};
		SimplePopulationImpl population3 = new SimplePopulationImpl(populationArray3);
		int[] expecteds3 = {0, 0, 0, 0, 0};
		int[] actuals3 = IDExogenousRulePayoffCalculator.unpack(population3);
		Assert.assertArrayEquals(expecteds3, actuals3);
		
		
	}
	
	//@Test
	public void testRepack() {
		//IDExogenousRulePayoffCalculator payoffCalculator = new IDExogenousRulePayoffCalculator(updateRule, chi, epsilon, alpha, a, b, c, d, institutionNumber, numberOfRounds)
		
		int[] extendedPop = {2, 2, 3, 0};
		double[] extendedFit = {1, 2, 3, 4};
		double[] actuals = IDExogenousRulePayoffCalculator.repack(extendedPop, extendedFit, 4);
		double[] expecteds = {4, 0, 1.5, 3};
		Assert.assertArrayEquals(expecteds, actuals, 0.001);
		
		int[] extendedPop2 = {0, 0, 1, 1, 0};
		double[] extendedFit2 = {1, 2, 3, 4, 5};
		double[] actuals2 = IDExogenousRulePayoffCalculator.repack(extendedPop2, extendedFit2, 2);
		double[] expecteds2 = {2.666666, 3.5};
		Assert.assertArrayEquals(expecteds2, actuals2, 0.001);
		
		
		
	}
	
	
	@Test
	public void testNOPayOffs(){
		Random.seed();
		UpdateRule updateRule = UpdateRule.OI;
		double chi = 0.01; //0.01;
		double epsilon = 0.08; //0.08;
		double alpha = 0.01; //0.01;
		double a = 1.0; //4.0;
		double b = -1.0; // -1.0;
		double c = 2.0; //3.0;
		double d = 0.0; // 0.0; 
		int institutionNumber = 152; //26
		int numberOfRounds = 100000; //100000
		//-----
		int numberOfMutants = 1;
		int popSize = 10;
		int indexResident = 2;
		int indexMutant = 0;
		int[] popArray = new int[4];
		popArray[indexResident] = popSize - numberOfMutants;
		popArray[indexMutant] = numberOfMutants;
		
		IDExogenousRulePayoffCalculator calculator = new IDExogenousRulePayoffCalculator(updateRule,chi, epsilon,alpha, a, b, c, d,institutionNumber, numberOfRounds);
		SimplePopulationImpl population = new SimplePopulationImpl(popArray);
		double[] ans = calculator.getPayoff(population);
		System.out.println("Mutant payoff " + ans[indexMutant]);
		System.out.println("Resident payoff " + ans[indexResident]);
		System.out.println("Mutant fitness " + Math.exp(ans[indexMutant]));
		System.out.println("Resident fitness " + Math.exp(ans[indexResident]));
		Assert.assertTrue(true);
	}
	


}
