#include<iostream>
#include<random>
#include<tuple>
#include <fstream>
#include<bitset>
#include<math.h>
#include <stdlib.h> /* for srand */
#include <string>
#include <iomanip>      // std::setprecision
#include <sstream> //string stream
using namespace std;


//sum array elements
int sum_array(int L[], int size) {
	int summ = 0;
	for (int i = 0; i < size; i++) {
		summ += L[i];
	}
	return summ;
}


//convert double to string with precision
string convert(double i) {
	stringstream stream;
	stream << fixed << setprecision(3) << i;
	string a = stream.str();
	return a;
}


double uni_rand() {
	mt19937 generator(rand());
	uniform_real_distribution <> dis(0.0, 1.0);
	double rd = dis(generator);
	return rd;
}


//random from a range [a£¬b]
int rand_int(int a, int b) {

	mt19937 generator(rand());
	uniform_int_distribution<> dis(a, b);
	int rd = dis(generator);
	return rd;
}




int rep_assignment(int d[], int c, int y) {
	int rep = 0;
	if (y == 1) {
		if (c == 1) {
			rep = d[0];
		}
		else {
			rep = d[1];
		}
	}
	else {
		if (c == 1) {
			rep = d[2];
		}
		else {
			rep = d[3];
		}
	}return rep;
}



pair<int, int> payoff(int x, int y, int P[], int D[], int N[], int benefit, int cost, double alpha, double eps, double ki, double tau) {
	//translate behaviour rules
	int p1[2] = {};
	int p2[2] = {};
	bitset<2>s1(P[x]);
	bitset<2>s2(P[y]);
	for (int i = 0; i < 2; i++) {
		p1[i] = s1[i];
		p2[i] = s2[i];
	}
	//pairwise
	int norm_x[4];
	int norm_y[4];
	bitset<4>s3(N[x]);
	bitset<4>s4(N[y]);
	for (int i = 0; i < 4; i++) {
		norm_x[i] = s3[i];
		norm_y[i] = s4[i];
	}
	int C_x;
	int C_y;
	if (uni_rand() < ki) {
		if (uni_rand() < eps && p1[1 - D[y]] == 1) {
			C_x = 0;
		}
		else {
			C_x = p1[1 - D[y]];
		}
	}
	else if (uni_rand() < eps && p1[D[y]] == 1) {
		C_x = 0;
	}
	else {
		C_x = p1[D[y]];
	}

	int x_rep = D[x];

	if (uni_rand() < tau) {
		if (uni_rand() < alpha) {
			D[x] = 1 - rep_assignment(norm_y, C_x, D[y]);
		}
		else {
			D[x] = rep_assignment(norm_y, C_x, D[y]);
		}
	}
	// same for y
	if (uni_rand() < ki) {
		if (uni_rand() < eps && p2[1 - x_rep] == 1) {
			C_y = 0;
		}
		else {
			C_y = p2[1 - x_rep];
		}
	}
	else if (uni_rand() < eps && p2[x_rep] == 1) {
		C_y = 0;
	}
	else {
		C_y = p2[x_rep];
	}
	if (uni_rand() < tau) {
		if (uni_rand() < alpha) {
			D[y] = 1 - rep_assignment(norm_x, C_y, x_rep);
		}
		else {
			D[y] = rep_assignment(norm_x, C_y, x_rep);
		}
	}
	int res;
	if (C_x == 1 && C_y == 1) {
		res = 0;
	}
	else if (C_x == 1) {
		res = 1;
	}
	else if (C_y == 1) {
		res = 2;
	}
	else {
		res = 3;
	}
	int value = benefit * C_y - cost * C_x;
	return make_pair(value, res);
}






int simulation(string name, int runs, int g, int z, int benefit, int cost, double eps, double alpha, double ki, double tau, double mu, int seed) {
	//file handling
	ofstream output;
	output.open(name);
	//seed
	srand(seed);

	//write header
	output << "run,time_step,a,b,c,d,avg_cooperation,mut/imi,%Good,";
	string type[4] = { "ALL_D","Disc","pDisc","All_C" };
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 16; j++) {
			output << type[i] + to_string(j) << ",";
		}
	}

	//%good
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 16; j++) {
			if (i == 3 && j == 15) {
				output << type[i] + to_string(j) + "%good" << endl;
			}
			else {
				output << type[i] + to_string(j) + "%good" << ",";
			}
		}
	}
	//dynamic array, first null ptr
	int* P = NULL;
	int* D = NULL;
	int* N = NULL;
	//new operator to allocate by size
	P = new int[z];
	D = new int[z];
	N = new int[z];
	for (int i = 0; i < z; i++) {
		P[i] = 0;
		D[i] = 0;
		N[i] = 0;
	}

	for (int r = 0; r < runs; r++) {
		for (int k = 0; k < z; k++) {
			P[k] = rand_int(0, 3);
			D[k] = rand_int(0, 1);
			N[k] = rand_int(0, 15);
		}
		//gen

		for (int t = 0; t < g; t++) {

			int coop = 0;
			int choices[4] = { 0 };
			int a = rand_int(0, z - 1);
			string state = "";
			if (uni_rand() < mu) {
				P[a] = rand_int(0, 3);
				state = "mutation";
			}
			else {
				state = "imitation";
				int b;
				do {
					b = rand_int(0, z - 1);

				} while (b == a);

				double fitness_a = 0;
				double fitness_b = 0;

				for (int i = 0; i < (2 * z); i++) {
					int c;

					do {
						c = rand_int(0, z - 1);
					} while (c == a);
					int val;
					int ct;
					tie(val, ct) = payoff(a, c, P, D, N,benefit, cost, alpha, eps, ki, tau);
					fitness_a += val;
					if (ct != 3) {
						if (ct == 0) {
							coop += 2;
						}
						else {
							coop += 1;
						}
					}
					choices[ct] += 1;
					//update b,c
					do {
						c = rand_int(0, z - 1);
					} while (c == b);

					tie(val, ct) = payoff(b, c, P, D, N, benefit, cost, alpha, eps, ki, tau);
					fitness_b += val;
					if (ct != 3) {
						if (ct == 0) {
							coop += 2;
						}
						else {
							coop += 1;
						}
					}
					choices[ct] += 1;

				}

				//fermi
				fitness_a /= (2 * z);
				fitness_b /= (2 * z);
				if (uni_rand() < 1 / (1 + exp(fitness_a - fitness_b))) {
					P[a] = P[b];
				}
			}

			double avg1;
			if (state == "imitation") {
				avg1 = (double)coop / (double)(8 * z);
			}
			else {
				avg1 = 0;
			}
			//csv
			int strat[64] = { 0 }; //# of each type
			int index;
			int j; //norm
			int G[64] = { 0 }; // # of Good for each of 64 strat
			for (int i = 0; i < z; i++) {
				j = N[i];
				if (P[i] == 0) {
					index = 0 + j;
				}
				else if (P[i] == 1) {
					index = 16 + j;
				}
				else if (P[i] == 2) {
					index = 32 + j;
				}
				else {
					index = 48 + j;
				}
				//update freq
				strat[index] += 1;

				//if good, add to G
				if (D[i] == 1) {
					G[index] += 1;
				}
			}



			output << r << "," << t << "," << choices[0] << "," << choices[1] << "," << choices[2] << "," << choices[3] << "," << avg1 << "," << state << "," << (float)sum_array(D, z) / (float)z << ",";

			//freq 
			for (int i = 0; i < 64; i++) {
				output << strat[i];
				output << ",";

			}
			//%good
			for (int i = 0; i < 64; i++) {
				if (G[i] != 0) {
					output << ((float)G[i] / (float)strat[i]);
				}
				else {
					output << 0;
				}

				if (i != 63) {
					output << ",";
				}

			}
			output << endl;


		}
	}

	output.close();
	return 0;

}

int main(int argc, char *argv[]) {



	int runs = atoi(argv[1]);
	int g = atoi(argv[2]);
	int z = atoi(argv[3]);
	int benefit = atoi(argv[4]);
	int cost = atoi(argv[5]);
	double eps = atof(argv[6]);
	double alpha = atof(argv[7]);
	double ki = atof(argv[8]);
	double tau = atof(argv[9]);
	double mu = 1 / atof(argv[10]);
	int seed = atoi(argv[11]);
	string name = "pw_runs_" + to_string(runs) + "_gen_" + to_string(g) + "_pop_" + to_string(z) + "_b_" + to_string(benefit) + "_c_" + to_string(cost) + "_eps_" + convert(eps) + "_alpha_" + convert(alpha) + "_ki_" + convert(ki) + "_tau_" + convert(tau) + "_mu_" + convert(mu) + "_seed_" + to_string(seed) + ".csv";

	simulation(name, runs, g, z, benefit, cost, eps, alpha, ki, tau, mu, seed);
	cout << "done";
	return 0;
}
