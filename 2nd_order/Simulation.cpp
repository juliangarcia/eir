#include<iostream>
#include<random>
#include<tuple>
#include <fstream>
#include<bitset>
#include<math.h>
#include <stdlib.h> // for srand 
#include <string>
#include <iomanip>      // std::setprecision
#include <sstream>      //string stream
using namespace std;

//sum array elements
int sum_array(int L[], int size) {
	int summ = 0;
	for (int i = 0; i < size; i++) {
		summ += L[i];
	}
	return summ;
}


//convert double to string with precision 3
string convert(double i) {
	stringstream stream;
	stream << fixed << setprecision(3) << i;
	string a = stream.str();
	return a;
}

//random in [0,1£©
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

//count to python
int count(int L[], int size, int target) {
	int ctr = 0;
	for (int i = 0; i < size; i++) {
		if (L[i] == target) {
			ctr += 1;
		}
	}
	return ctr;
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
pair<int, int> payoff(int x, int y, int norm[], int P[], int D[], int benefit, int cost, double alpha, double eps, double ki, double tau) {
	//translate behaviour rules, for convenience P(pr B, pr G)
	int p1[2] = {};
	int p2[2] = {};
	bitset<2>s1(P[x]);
	bitset<2>s2(P[y]);
	for (int i = 0; i < 2; i++) {
		p1[i] = s1[i];
		p2[i] = s2[i];
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
	int x_rep = D[x]; // rep could change for x, y considers x's old rep
	if (uni_rand() < tau) {
		if (uni_rand() < alpha) {
			D[x] = 1 - rep_assignment(norm, C_x, D[y]);
		}
		else {
			D[x] = rep_assignment(norm, C_x, D[y]);
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
			D[y] = 1 - rep_assignment(norm, C_y, x_rep);
		}
		else {
			D[y] = rep_assignment(norm, C_y, x_rep);
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
int simulation(string name, int runs, int g, int z, int norm[], int benefit, int cost, double eps, double alpha, double ki, double tau, double mu, int seed) {
	srand(seed);
	//file handling
	ofstream output;
	output.open(name);
	//write header
	output << "run,time_step,ALL_D,ALL_D%Good,Disc,Disc%Good,pDisc,pDisc%Good,ALL_C,ALL_C%Good,a,b,c,d,avg_cooperation,mut/imi,%Good,\n";
	//dynamic array, first null ptr
	int* P = NULL;
	int* D = NULL;
	//new operator to allocate by size
	P = new int[z];
	D = new int[z];
	for (int i = 0; i < z; i++) {
		P[i] = 0;
		D[i] = 0;
	}
	for (int r = 0; r < runs; r++) {
		for (int k = 0; k < z; k++) {
			P[k] = rand_int(0, 3);
			D[k] = rand_int(0, 1);
		}
		//gen
		for (int t = 0; t < g; t++) {
			int coop = 0;
			int choices[4] = { 0,0,0,0 };
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
				for (int i = 0; i < 2 * z; i++) {
					int c;
					do {
						c = rand_int(0, z - 1);
					} while (c == a);
					int val;
					int ct;
					tie(val, ct) = payoff(a, c, norm, P, D, benefit, cost, alpha, eps, ki, tau);
					fitness_a += val;
					if (ct != 3) {
						if (ct == 0) {
							coop += 2;
						}
						else {
							coop += 1;
						}
					}
					//update 4 outcomes{a,b,c,d}
					choices[ct] += 1;
					//update b,c
					do {
						c = rand_int(0, z - 1);
					} while (c == b);
					tie(val, ct) = payoff(b, c, norm, P, D, benefit, cost, alpha, eps, ki, tau);
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


			//calculate %Good
			int F[4] = {0,0,0,0}; // total freq of each type
			int G[4] = {0,0,0,0};// total Good individuals of each type
			for (int i = 0; i < z; i++) {
				F[P[i]] += 1;
				if (D[i] == 1) {
					G[P[i]] += 1;
				}
			}

			double prop[4] = {};
			for (int i = 0; i < 4; i++) {
				if (F[i] != 0) {
					prop[i] = (float) G[i] / (float) F[i];
				}
				else {
					prop[i] = 0.0;
				}
			}

			//csv  run,time_step,ALL_D,ALL_D%Good,Disc,Disc%Good,pDisc,pDisc%Good,ALL_C,ALL_C%Good,
			output << r << ',' << t << ',' << F[0] << ',' << prop[0] << ',' << F[1] << ',' << prop[1] << ',' << F[2] << ',' << prop[2] << ',' << F[3] << ',' << prop[3] << ',' << choices[0] << ',' << choices[1] << ',' << choices[2] << ',' << choices[3] << ',' << avg1 << ',' << state << ',' << (float)sum_array(G, 4) / (float)z << endl;
		}
	}
	output.close();
	return 0;
}
int main(int argc, char *argv[]) {
	//if (argc != 11) return -1;
	int runs = atoi(argv[1]);
	int g = atoi(argv[2]);
	int z = atoi(argv[3]);
	int norm_dec = atoi(argv[4]);
	int benefit = atoi(argv[5]);
	int cost = atoi(argv[6]);
	double eps = atof(argv[7]);
	double alpha = atof(argv[8]);
	double ki = atof(argv[9]);
	double tau = atof(argv[10]);
	double mu = 1 / atof(argv[11]);       //takes in 1/input(int)
	int seed = atoi(argv[12]);
	int norm[4] = {};
	bitset<4>ss(norm_dec);
	for (int i = 0; i < 4; i++) {
		norm[i] = ss[i];
	}
	string name = "og_runs_" + to_string(runs) + "_gen_" + to_string(g) + "_pop_" + to_string(z) + "_norm_" + to_string(norm_dec) + "_b_" + to_string(benefit) + "_c_" + to_string(cost) + "_eps_" + convert(eps) + "_alpha_" + convert(alpha) + "_ki_" + convert(ki) + "_tau_" + convert(tau) + "_mu_" + convert(mu) + "_seed_" + to_string(seed) + ".csv";


	simulation(name, runs, g, z, norm, benefit, cost, eps, alpha, ki, tau, mu, seed);
	cout << "done";
	return 0;
}
