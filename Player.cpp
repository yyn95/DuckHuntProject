#include "Player.hpp"
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <time.h>

namespace ducks {


    std::vector<ESpecies> allSpecies = {SPECIES_PIGEON, SPECIES_RAVEN, SPECIES_SKYLARK, SPECIES_SWALLOW, SPECIES_SNIPE,
                                        SPECIES_BLACK_STORK};

    std::vector<EMovement> allMovements = {MOVE_UP_LEFT,
                                           MOVE_UP,
                                           MOVE_UP_RIGHT,
                                           MOVE_LEFT,
                                           MOVE_STOPPED,
                                           MOVE_RIGHT,
                                           MOVE_DOWN_LEFT,
                                           MOVE_DOWN,
                                           MOVE_DOWN_RIGHT};

    std::vector<std::vector<double **> > bird_models(6, std::vector<double **>(0));
    std::vector<std::vector<Bird>> movs_hist(7, std::vector<Bird>(0));
    std::vector<std::vector<Bird>> movs_tolearn(6, std::vector<Bird>(0));

    size_t max_obs = 1000;

    double prob_shooting = 0.7;
    size_t min_temp_shoot = 80;
    size_t min_known = 400;
    size_t starting_round = 3;

    size_t shoots = 0;
    size_t dead_birds = 0;

    size_t knowleadge[6];
    size_t learned[6];

    std::vector<double **> A_noms(6);
    std::vector<double **> A_dens(6);
    std::vector<double **> B_noms(6);
    std::vector<double **> B_dens(6);
    std::vector<double **> pi_unnormalized(6);

    int fail_train = 0;

    size_t total_knowleadge = 0;
    size_t knowleadge_guess = 30;



    Player::Player() {}

    void sleep(unsigned int mseconds) {
        clock_t goal = mseconds + clock();
        while (goal > clock());
    }

    void display(size_t nrow, size_t ncol, double ***mat) {

        std::cerr << "    ";

        for (size_t i = 0; i < nrow; i++) {

            for (size_t j = 0; j < ncol; j++) {

                std::cerr << round((*mat)[i][j] * 1000.) / 1000. << " ";
            }

            std::cerr << std::endl << "    ";
        }
    }


    void seq2model(size_t N, size_t K, size_t T, int **obs, double ***A_old, double ***B_old, double **pi_old,
                   double ***A_nom, double ***A_den, double ***B_nom, double ***B_den) {

        srand(time(NULL));

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {

                (*A_old)[i][j] = 10000. + (double) rand() / RAND_MAX;
            }

            for (size_t k = 0; k < K; k++) {
                (*B_old)[i][k] = 10000. + (double) rand() / RAND_MAX;
            }

            (*pi_old)[i] = 10000. + (double) rand() / RAND_MAX;

        }

        //Normalization of the values and duplicates in Matrix_new

        double A_new[N][N];
        double B_new[N][K];
        double pi_new[N];

        double sum_pi = 0.;

        for (size_t i = 0; i < N; i++) {
            double sum = 0.;

            for (size_t j = 0; j < N; j++) {
                sum += (*A_old)[i][j];
            }

            for (size_t j = 0; j < N; j++) {
                (*A_old)[i][j] /= sum;
                A_new[i][j] = (*A_old)[i][j];
            }

            sum = 0.;

            for (size_t k = 0; k < K; k++) {
                sum += (*B_old)[i][k];
            }

            for (size_t k = 0; k < K; k++) {
                (*B_old)[i][k] /= sum;
                B_new[i][k] = (*B_old)[i][k];
            }

            sum_pi += (*pi_old)[i];

        }
        for (size_t i = 0; i < N; i++) {
            (*pi_old)[i] /= sum_pi;
            pi_new[i] = (*pi_old)[i];
        }

        // N = K
//
//        for(size_t i = 0; i < N; i++){
//            for(size_t j = 0; j < N; j++){
//
//                if(i == j){
//                    (*B_old)[i][j] = 1.;
//                    B_new[i][j] = 1.;
//                    (*A_old)[i][j] = 1.;
//                    A_new[i][j] = 1.;
//                } else{
//                    (*B_old)[i][j] = 0.;
//                    B_new[i][j] = 0.;
//                    (*A_old)[i][j] = 0.;
//                    A_new[i][j] = 0.;
//
//                }
//            }
//        }

        double alfa[T][N];
        double beta[T][N];
        double gammaij[T - 1][N][N];
        double gamma[T][N];

        // main program, calculation of A, B and pi from Observation sequence

        //For make it easy the calculation of B, we will make the obs_ts
        //obs_ts[j] = vector with the time positions where emission j appears

        std::vector<std::vector<size_t>> obs_ts;

        for (int j = 0; j < K; j++) {
            std::vector<size_t> dummie;
            obs_ts.push_back(dummie);
        }

        for (size_t t = 0; t < T; t++) {

            if ((*obs)[t] == -1) {
                T = t;
                break;
            }

            obs_ts[(*obs)[t]].push_back(t);
        }

        //intialize counter "cont", the log-probabilities and scale_c, where we will save the sclae constants

        size_t cont = 0, max_iter = 3000;

        double scale_c[T], new_log_prob = 1., old_log_prob = 1.;

        //std::cerr << std::endl << "Matrix A before bucle:" << std::endl << std::endl;
        //display(N, N, &(*A_old));

        //std::cerr << std::endl << "Matrix B before bucle:" << std::endl << std::endl;
        //display(N, K, &(*B_old));

        do {

            cont++;

            //Update matrix

            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    (*A_old)[i][j] = A_new[i][j];
                }
                for (size_t k = 0; k < N; k++) {
                    (*B_old)[i][k] = B_new[i][k];
                }
                (*pi_old)[i] = pi_new[i];
            }

            //Calculate new alfa and scale parameters

            //alfa time-0
            scale_c[0] = 0.;

            for (int i = 0; i < N; i++) {
                alfa[0][i] = (*pi_old)[i] * (*B_old)[i][(*obs)[0]];
                scale_c[0] += alfa[0][i];
            }
            scale_c[0] = 1. / scale_c[0];

            for (int i = 0; i < N; i++) {
                alfa[0][i] *= scale_c[0];
            }

            //alfa time 1 to T-1

            for (size_t t = 1; t < T; t++) {
                scale_c[t] = 0.;

                for (int i = 0; i < N; i++) {

                    //Sum term

                    alfa[t][i] = 0;

                    for (int j = 0; j < N; j++) {
                        alfa[t][i] += alfa[t - 1][j] * (*A_old)[j][i];
                    }

                    alfa[t][i] *= (*B_old)[i][(*obs)[t]];
                    scale_c[t] += alfa[t][i];
                }

                scale_c[t] = 1. / scale_c[t];

                for (int i = 0; i < N; i++) {
                    alfa[t][i] *= scale_c[t];
                }

            }

            //Calculate new Beta

            //beta time T-1

            for (int i = 0; i < N; i++) {
                beta[T - 1][i] = scale_c[T - 1];
            }

            //beta time T-2 to 0

            for (long int t = T - 2; t >= 0; t--) {

                for (int i = 0; i < N; i++) {

                    //sum term

                    beta[t][i] = 0.;


                    for (int j = 0; j < N; j++) {
                        beta[t][i] += beta[t + 1][j] * (*A_old)[i][j] * (*B_old)[j][(*obs)[t + 1]];
                    }

                    beta[t][i] *= scale_c[t];

                }
            }

            // Both gammas are calculated at the same time


            for (long int t = 0; t < (long int) T - 1; t++) {

                double norm = 0.;

                for (int i = 0; i < N; i++) {

                    for (int j = 0; j < N; j++) {
                        norm += alfa[t][i] * (*A_old)[i][j] *
                                (*B_old)[j][(*obs)[t + 1]] * beta[t + 1][j];

                    }

                }

                for (int i = 0; i < N; i++) {

                    gamma[t][i] = 0.;

                    for (int j = 0; j < N; j++) {
                        gammaij[t][i][j] = alfa[t][i] * (*A_old)[i][j] *
                                           (*B_old)[j][(*obs)[t + 1]] * beta[t + 1][j] / norm;
                        gamma[t][i] += gammaij[t][i][j];
                    }

                }

            }

            //for t = T-1

            double norm = 0.;
            for (int i = 0; i < N; i++) {
                norm += alfa[T - 1][i];
            }

            for (int i = 0; i < N; i++) {
                gamma[T - 1][i] = alfa[T - 1][i] / norm;
            }



            //Calculate news A, B and Pi matrix

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {

                    double aij = 0.;
                    double norm = 0.;

                    for (size_t t = 0; t < T - 1; t++) {
                        aij += gammaij[t][i][j];
                        norm += gamma[t][i];
                    }

                    A_new[i][j] = (double) (aij / norm);
                }
            }


            for (int i = 0; i < N; i++) {

                double norm = 0.;

                for (size_t t = 0; t < T; t++) {
                    norm += gamma[t][i];
                }


                for (int k = 0; k < K; k++) {

                    double bki = 0.;

                    for (size_t l = 0; l < obs_ts[k].size(); l++) {
                        bki += gamma[obs_ts[k][l]][i];
                    }

                    B_new[i][k] = (double) (bki / norm);

                }


            }


            for (int i = 0; i < N; i++) {
                pi_new[i] = (double) gamma[0][i];
            }

            //Calculate prob of the new model

            old_log_prob = new_log_prob;

            new_log_prob = 0.;

            for (size_t t = 0; t < T; t++) {
                new_log_prob += std::log(scale_c[t]);
            }

            new_log_prob = -new_log_prob;

            if (cont == 1) {
                old_log_prob = new_log_prob - 1.;
            }

            if (!(cont % 100)) {
                std::cerr << cont << std::endl;
            }

        } while (cont < max_iter && new_log_prob > old_log_prob);

        if (cont != max_iter) {

            //Calculate news A, B nominators and denominators

            for (int i = 0; i < N; i++) {

                for (int j = 0; j < N; j++) {

                    double aij = 0.;
                    double norm = 0.;

                    for (size_t t = 0; t < T - 1; t++) {

                        aij += gammaij[t][i][j];
                        norm += gamma[t][i];

                    }

                    (*A_nom)[i][j] = aij;
                    (*A_den)[i][j] = norm;
                }

            }


            for (int i = 0; i < N; i++) {

                double norm = 0.;

                for (size_t t = 0; t < T; t++) {
                    norm += gamma[t][i];
                }

                for (int k = 0; k < K; k++) {

                    double bki = 0.;

                    for (size_t l = 0; l < obs_ts[k].size(); l++) {
                        bki += gamma[obs_ts[k][l]][i];
                    }

                    (*B_nom)[i][k] = bki;
                    (*B_den)[i][k] = norm;

                }

            }
        } else {
            fail_train = 1;
        };

        //std::cerr << std::endl << std::endl << "Num of iterations : " << cont << std::endl;
        //std::cerr << std::endl << std::endl << std::endl;
        //std::cerr << std::endl << "Matrix A after bucle:" << std::endl << std::endl;
        //display(N, N, &(*A_old));

        //std::cerr << std::endl << "Matrix B after bucle:" << std::endl << std::endl;
        //display(N, K, &(*B_old));

    }


    void add_arr(double ***A, double ***B, size_t nrow, size_t ncol) {

        for (size_t i = 0; i < nrow; i++) {

            for (size_t j = 0; j < ncol; j++) {
                (*A)[i][j] += (*B)[i][j];
            }

        }

    }

    void copy_arr(double ***A, double ***B, size_t nrow, size_t ncol) {

        for (size_t i = 0; i < nrow; i++) {

            for (size_t j = 0; j < ncol; j++) {
                (*A)[i][j] = (*B)[i][j];
            }

        }

    }

    void divide_arr(double ***A, double ***B, size_t nrow, size_t ncol) {

        for (size_t i = 0; i < nrow; i++) {

            for (size_t j = 0; j < ncol; j++) {
                (*A)[i][j] /= (*B)[i][j];
            }

        }

    }

    void norm_arr(double ***A, size_t nrow, size_t ncol) {

        for (size_t i = 0; i < nrow; i++) {

            double sum = 0.;

            for (size_t j = 0; j < ncol; j++) {
                sum += (*A)[i][j];
            }

            for (size_t j = 0; j < ncol; j++) {
                (*A)[i][j] /= sum;
            }

        }

    }

    void bb_models() {

        for (size_t i = 0; i < bird_models.size(); i++) {

            //build observation sequence

            size_t T = 0;

            for (size_t j = 0; j < movs_tolearn[i].size(); j++) {
                T += (size_t) movs_tolearn[i][j].getSeqLength();
            }

            if (T > max_obs) {
                T = max_obs;
            }

            if (T > 0) {

                int *obs = new int[T];

                size_t cont = 0;

                for (size_t j = 0; j < movs_tolearn[i].size(); j++) {

                    for (size_t k = 0; k < movs_tolearn[i][j].getSeqLength(); k++) {

                        obs[cont] = movs_tolearn[i][j].getObservation((int) k);

                        if (obs[cont] == -1) {
                            break;
                        }
                        cont++;

                        if (cont == max_obs) {
                            std::cerr << "exit" << std::endl;
                            goto exit;
                        }

                    }

                }

                T = cont;

                exit:

                //Num Hidden states (Patterns)

                size_t N = 5;

                //Num observations (movements)

                size_t K = 9;

                auto **A = new double *[N];
                auto **B = new double *[N];
                auto **A_nom = new double *[N];
                auto **A_den = new double *[N];
                auto **B_nom = new double *[N];
                auto **B_den = new double *[N];
                auto *pi = new double[N];

                for (size_t i = 0; i < N; i++) {
                    A[i] = new double[N];
                    B[i] = new double[K];
                    A_nom[i] = new double[N];
                    A_den[i] = new double[N];
                    B_nom[i] = new double[K];
                    B_den[i] = new double[K];
                }

                seq2model(N, K, T, &obs, &A, &B, &pi, &A_nom, &A_den, &B_nom, &B_den);

                if (fail_train == 0) {

                    learned[i] = T;

                    //update A_noms, A_dens, B_noms, B_dens

                    add_arr(&(A_noms[i]), &A_nom, N, N);
                    add_arr(&(A_dens[i]), &A_den, N, N);

                    copy_arr(&A, &(A_noms[i]), N, N);

                    divide_arr(&A, &(A_dens[i]), N, N);

                    add_arr(&(B_noms[i]), &B_nom, N, K);
                    add_arr(&(B_dens[i]), &B_den, N, K);

                    copy_arr(&B, &(B_noms[i]), N, K);

                    divide_arr(&B, &(B_dens[i]), N, K);

                    //Update model

                    norm_arr(&A, N, N);
                    norm_arr(&B, N, K);

                    auto **pi_new = new double *[0];

                    pi_new[0] = pi;

                    add_arr(&(pi_unnormalized[i]), &pi_new, 1, N);

                    copy_arr(&pi_new, &(pi_unnormalized[i]), 1, N);

                    norm_arr(&pi_new, 1, N);

                    bird_models[i].clear();

                    bird_models[i].push_back(A);
                    bird_models[i].push_back(B);
                    bird_models[i].push_back(pi_new);

                    movs_tolearn[i].clear();

                    //display(N, N, &A);
                    //display(N, K, &B);

                    //display(N, N, &A_noms[i]);
                    //display(N, K, &A_dens[i]);
                } else {

                    //std::cerr<< "HMM training didn't converge!" << std::endl;
                    fail_train = 0;

                }

            }
        }

    }

    void train_MM(double ***A, int *obs, size_t K, size_t T) {

        for (size_t i = 0; i < K; i++) {
            for (size_t j = 0; j < K; j++) {
                (*A)[i][j] = 0.;
            }
        }

        for (size_t t = 0; t < T - 1; t++) {
            (*A)[obs[t]][obs[t + 1]] += 1.;
        }

        norm_arr(A, K, K);

    }

    void train_MM2(double ***A, int *obs, size_t K, size_t T) {

        for (size_t i = 0; i < K * K; i++) {
            for (size_t j = 0; j < K * K; j++) {
                (*A)[i][j] = 0.;
            }
        }

        for (size_t t = 0; t < T - 2; t++) {
            (*A)[obs[t] * allMovements.size() + obs[t + 1]][obs[t + 1] * allMovements.size() + obs[t + 2]] += 1.;
        }

        norm_arr(A, K * K, K * K);

    }

    void add_MM(double ***A, int *obs, size_t K, size_t T) {

        for (size_t t = 0; t < T - 1; t++) {
            (*A)[obs[t]][obs[t + 1]] += 1.;
        }

    }


    void mseq2model(Bird bird, double ***A, double ***B, double **pi) {

        auto T = (size_t) bird.getSeqLength();

        int *obs = new int[T];

        for (size_t i = 0; i < T; i++) {

            obs[i] = bird.getObservation((int) i);

        }

        //Num Hidden states (Patterns)

        size_t N = 5;

        //Num observations (movements)

        size_t K = 9;

        (*A) = new double *[N];
        (*B) = new double *[N];
        (*pi) = new double[N];

        for (size_t i = 0; i < N; i++) {
            (*A)[i] = new double[N];
            (*B)[i] = new double[K];
        }

        //seq2model(N, K, T, &obs, &(*A), &(*B), &(*pi));

/*

        std::cerr << std::endl << std::endl << "A Matrix" << std::endl << std::endl;
        display(N,N,&A);

        std::cerr << std::endl << std::endl << "B Matrix" << std::endl << std::endl;
        display(N,K,&B);

*/

    }

    double distance(double **A1, double **B1, double **A2, double **B2, size_t N, size_t K) {

        double distance = 0.;

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                distance += (A1[i][j] - A2[i][j]) * (A1[i][j] - A2[i][j]);
            }

            for (size_t k = 0; k < K; k++) {
                distance += (B1[i][k] - B2[i][k]) * (B1[i][k] - B2[i][k]);
            }
        }


        return sqrt(distance);

    }


    double loglikelihood(double ***A, double ***B, double **pi, Bird bird, size_t N, size_t K) {

        auto T = (size_t) bird.getSeqLength();

        auto *obs = new int[T];

        for (size_t i = 0; i < T; i++) {

            obs[i] = bird.getObservation((int) i);

            if (obs[i] == -1) {
                T = i;
                break;
            }
        }

        //Calculate the log likelihood from alfas

        double alfa[T][N];
        double scale_c[T];

        //Calculate new alfa and scale parameters

        //alfa time-0
        scale_c[0] = 0.;

        for (int i = 0; i < N; i++) {
            alfa[0][i] = (*pi)[i] * (*B)[i][obs[0]];
            scale_c[0] += alfa[0][i];
        }
        if (scale_c[0] == 0) {

            return -std::numeric_limits<double>::infinity();

        }
        scale_c[0] = 1. / scale_c[0];

        for (int i = 0; i < N; i++) {
            alfa[0][i] *= scale_c[0];
        }

        //alfa time 1 to T-1

        for (size_t t = 1; t < T; t++) {
            scale_c[t] = 0.;

            for (int i = 0; i < N; i++) {

                //Sum term

                alfa[t][i] = 0;

                for (int j = 0; j < N; j++) {

                    alfa[t][i] += alfa[t - 1][j] * (*A)[j][i];

                }

                alfa[t][i] *= (*B)[i][obs[t]];
                scale_c[t] += alfa[t][i];
            }
/*
            if (scale_c[t] == 0.) {
                std::cerr << "parametro c es cero   " << t << "   " << T << std::endl;
                std::cerr << "alfas   " << (*B)[0][obs[t]] << "   " << (*B)[1][obs[t]] << "   " << (*B)[2][obs[t]]
                          << "   " << (*B)[3][obs[t]] << "   " << (*B)[4][obs[t]] << "   " << std::endl;
            }*/
            if (scale_c[t] == 0) {

                return -std::numeric_limits<double>::infinity();

            }
            scale_c[t] = 1. / scale_c[t];

            for (int i = 0; i < N; i++) {
                alfa[t][i] *= scale_c[t];

            }

        }

        double log_prob = 0.;

        for (size_t t = 0; t < T; t++) {
            log_prob += std::log(scale_c[t]);
        }

        log_prob = -log_prob;
        return log_prob;

    }


    double *get_prob_emission(double **A, double **B, double *pi, Bird bird, size_t N, size_t K) {
        auto T = (size_t) bird.getSeqLength();

        auto *obs = new int[T];

        for (size_t i = 0; i < T; i++) {

            obs[i] = bird.getObservation((int) i);
        }


        double alfa[T][N];
        double beta[T][N];
        double gammaij[T - 1][N][N];
        double gamma[T][N];
        double scale_c[T];


        //Calculate new alfa and scale parameters

        //alfa time-0
        scale_c[0] = 0.;

        for (int i = 0; i < N; i++) {
            alfa[0][i] = pi[i] * B[i][obs[0]];
            scale_c[0] += alfa[0][i];
        }
        scale_c[0] = 1. / scale_c[0];

        for (int i = 0; i < N; i++) {
            alfa[0][i] *= scale_c[0];
        }


        //alfa time 1 to T-1

        for (size_t t = 1; t < T; t++) {
            scale_c[t] = 0.;

            for (int i = 0; i < N; i++) {

                //Sum term

                alfa[t][i] = 0;

                for (int j = 0; j < N; j++) {

                    alfa[t][i] += alfa[t - 1][j] * A[j][i];

                }

                alfa[t][i] *= B[i][obs[t]];
                scale_c[t] += alfa[t][i];
            }

            scale_c[t] = 1. / scale_c[t];

            for (int i = 0; i < N; i++) {
                alfa[t][i] *= scale_c[t];
            }

        }

        //Calculate new Beta

        //beta time T-1

        for (int i = 0; i < N; i++) {
            beta[T - 1][i] = scale_c[T - 1];
        }

        //beta time T-2 to 0

        for (long int t = T - 2; t >= 0; t--) {

            for (int i = 0; i < N; i++) {

                //sum term

                beta[t][i] = 0.;


                for (int j = 0; j < N; j++) {

                    beta[t][i] += beta[t + 1][j] * A[i][j] * B[j][obs[t + 1]];
                }

                beta[t][i] *= scale_c[t];

            }
        }

        // Both gammas are calculated at the same time

        for (long int t = 0; t < (long int) T - 1; t++) {

            double norm = 0.;

            for (int i = 0; i < N; i++) {

                for (int j = 0; j < N; j++) {

                    norm += alfa[t][i] * A[i][j] *
                            B[j][obs[t + 1]] * beta[t + 1][j];

                }

            }

            for (int i = 0; i < N; i++) {

                gamma[t][i] = 0.;

                for (int j = 0; j < N; j++) {

                    gammaij[t][i][j] = alfa[t][i] * A[i][j] *
                                       B[j][obs[t + 1]] * beta[t + 1][j] / norm;

                    gamma[t][i] += gammaij[t][i][j];

                }

            }

        }

        //for t = T-1

        double norm = 0.;
        for (int i = 0; i < N; i++) {
            norm += alfa[T - 1][i];
        }

        for (int i = 0; i < N; i++) {
            gamma[T - 1][i] = alfa[T - 1][i] / norm;
        }

        //calculate probabilities of emissions in next time-step

        auto *probs = new double[K];

        for (size_t k = 0; k < K; k++) {

            probs[k] = 0.;

            for (size_t i = 0; i < N; i++) {

                double sum = 0.;

                for (size_t j = 0; j < N; j++) {

                    sum += A[j][i] * gamma[T - 1][j];

                }
                probs[k] += sum * B[i][k];

            }
        }

        return probs;

    }


    Action Player::shoot(const GameState &pState, const Deadline &pDue) {
        /*
         * Here you should write your clever algorithms to get the best action.
         * This skeleton never shoots.
         */


        auto starting_timestep = min_temp_shoot;


        if (pState.getRound() < starting_round || pState.getBird(0).getSeqLength() < starting_timestep) {

            return cDontShoot;
        }


        double best_prob = 0.;

        Action best_action(0, MOVE_UP_LEFT);


        size_t N = 5;
        size_t K = 9;

        for (size_t i = 0; i < pState.getNumBirds(); i++) {

            double black_bird_likelihood;

            if (pState.getBird(i).getObservation(pState.getBird(i).getSeqLength() - 1) != -1) {//the bird is alive

                std::vector<double> likelihoods;


                for (size_t j = 0; j < 6; j++) {

                    if (!bird_models[j].empty() && knowleadge[j] > min_known) {

                        double lprob = loglikelihood(&bird_models[j][0], &bird_models[j][1], &bird_models[j][2][0],
                                                     pState.getBird((int) i), N, K);

                        likelihoods.push_back(lprob);

                    } else {
                        likelihoods.push_back(-1. / 0);
                    }

                }

                std::vector<int> ordered_species(likelihoods.size());
                std::size_t n(0);
                std::generate(std::begin(ordered_species), std::end(ordered_species), [&] { return n++; });

                std::sort(std::begin(ordered_species),
                          std::end(ordered_species),
                          [&](int i1, int i2) { return likelihoods[i1] < likelihoods[i2]; });

                //for(size_t b = 0; b < 6; b++){
                //    std::cerr << b << "   " << likelihoods[b] << std::endl;
                //}

                //std::cerr << "sorted vec" << std::endl;

                //for(size_t b = 0; b < 6; b++){
                //    std::cerr << ordered_species[b] << "   " << likelihoods[ordered_species[b]] << std::endl;
                //}

                if (ordered_species[5] != 5 && ordered_species[4] != 5) {
                    //bad bird

                    if (likelihoods[ordered_species[5]] > std::log(100.) + likelihoods[ordered_species[4]] &&
                        !std::isnan(likelihoods[ordered_species[5]])) {

                        //Not hidden markov model

                        auto **A = new double *[K];

                        for (size_t k = 0; k < K; k++) {
                            A[k] = new double[K];

                            for (size_t k2 = 0; k2 < K; k2++) {
                                A[k][k2] = 0.;
                            }
                        }

                        for (size_t j = 0; j < movs_hist[ordered_species[5]].size(); j++) {

                            auto T = (size_t) movs_hist[ordered_species[5]][j].getSeqLength();

                            auto *obs = new int[T];

                            for (size_t l = 0; l < T; l++) {

                                obs[l] = movs_hist[ordered_species[5]][j].getObservation((int) l);
                                if (obs[l] == -1) {
                                    T = l;
                                    break;
                                }

                            }
                            add_MM(&A, obs, K, T);

                        }

                        auto T = (size_t) pState.getBird(i).getSeqLength();

                        auto *obs = new int[T];

                        for (size_t l = 0; l < T; l++) {

                            obs[l] = pState.getBird(i).getObservation((int) l);
                            if (obs[l] == -1) {
                                T = l;
                                break;
                            }

                        }
                        add_MM(&A, obs, K, T);

                        norm_arr(&A, K, K);

                        //display(K, K, &A);

                        //display(K, K, &A);

                        //auto probs_mov = get_prob_emission(model[0], model[1], model[2][0], pState.getBird((int) i), N, K);

                        auto probs_mov = A[obs[T - 1]];

                        double max_prob = 0.;
                        int mov;

                        for (size_t j = 0; j < K; j++) {

                            if (probs_mov[j] > max_prob) {
                                max_prob = probs_mov[j];
                                mov = (int) j;

                            }
                        }

                        if (max_prob > best_prob) {
                            best_prob = max_prob;
                            best_action = Action((int) i, allMovements[mov % allMovements.size()]);
                        }

                        //if(best_prob > prob_shooting){
                        //    std::cerr << "shooting for row "<< mov << std::endl;
                        //    display(K, K, &A);
                        //}

                    }
                }


            }
        }

        if (best_prob > prob_shooting) {
            shoots++;
            std::cerr << "shooting!" << std::endl;
            return best_action;
        }

        return cDontShoot;

        //This line would predict that bird 0 will move right and shoot at it
        //return Action(0, MOVE_RIGHT);
    }

    std::vector<ESpecies> Player::guess(const GameState &pState, const Deadline &pDue) {
        /*
         * Here you should write your clever algorithms to guess the species of each bird.
         * This skeleton makes no guesses, better safe than sorry!
         */

        size_t N = 5;
        size_t K = 9;

        std::vector<ESpecies> lGuesses;
		
        if (total_knowleadge == 0) {

            for (size_t k = 0; k < 6; k++) {

                auto **A_nom = new double *[N];
                auto **A_den = new double *[N];
                auto **B_nom = new double *[N];
                auto **B_den = new double *[N];
                auto **pi = new double *[1];

                pi[0] = new double [N];

                for (size_t i = 0; i < N; i++) {

                    A_nom[i] = new double[N];
                    A_den[i] = new double[N];
                    B_nom[i] = new double[K];
                    B_den[i] = new double[K];

                    for (size_t j = 0; j < N; j++) {

                        A_nom[i][j] = 0.;
                        A_den[i][j] = 0.;
                    }

                    for (size_t j = 0; j < K; j++) {

                        B_nom[i][j] = 0.;
                        B_den[i][j] = 0.;
                    }

                }

                A_dens[k] = A_den;
                A_noms[k] = A_nom;
                B_dens[k] = B_den;
                B_noms[k] = B_nom;
                pi_unnormalized[k] = pi;

            }


            for (size_t i = 0; i < 6; i++) {
                knowleadge[i] = 0;
            }

            for (size_t i = 0; i < pState.getNumBirds(); i++) {
                lGuesses.push_back(allSpecies[0]);
            }

        } else {


            size_t temp_knowleadge[6];
            for (size_t i = 0; i < 6; i++) {
                temp_knowleadge[i] = knowleadge[i];
            }

            for (size_t i = 0; i < pState.getNumBirds(); i++) {

                //std::cerr << std::endl;

                double max_lprob = -1e100, second_lprob = -2e100;
                size_t max_indx = 0, second_indx = 0;
                std::vector<size_t> forbidden_birds;

                for (size_t j = 0; j < 6; j++) {
                    if (!bird_models[j].empty()) {

                        double lprob = loglikelihood(&bird_models[j][0], &bird_models[j][1], &bird_models[j][2][0],
                                                     pState.getBird((int) i), N, K);

                        if (lprob > max_lprob) {
                            second_lprob = max_lprob;
                            max_lprob = lprob;
                            second_indx = max_indx;
                            max_indx = j;
                        } else if (lprob > second_lprob) {
                            second_lprob = lprob;
                            second_indx = j;
                        }

                        if (std::isinf(lprob)) {
                            forbidden_birds.push_back(j);
                        }

                    }

                }

                if (!std::isnan(max_lprob)) {

                    lGuesses.push_back(allSpecies[max_indx]);

                } else if (total_knowleadge < knowleadge_guess) {
                    
                    size_t min_know_idx = 0, min_know = 1000000000;

                    for (size_t j = 0; j < 6; j++) {

                        if (std::find(forbidden_birds.begin(), forbidden_birds.end(), j) == forbidden_birds.end()) {
                            if (temp_knowleadge[i] < min_know) {
                                min_know = temp_knowleadge[i];
                                min_know_idx = j;
                            }
                        }

                    }

                    temp_knowleadge[min_know_idx]++;

                    lGuesses.push_back(allSpecies[min_know_idx]);

                } else {
                    std::cerr << "No guessing" << std::endl;
                    lGuesses.push_back(SPECIES_UNKNOWN);
                }

            }

        }

        //std::cerr << "Guesses" << std::endl << std::endl;

        //std::cerr << std::endl << std::endl;

        return lGuesses;
    }

    void Player::hit(const GameState &pState, int pBird, const Deadline &pDue) {
        /*
         * If you hit the bird you are trying to shoot, you will be notified through this function.
         */
        std::cerr << "HIT BIRD!!!" << std::endl;
        dead_birds++;
    }

    void Player::reveal(const GameState &pState, const std::vector<ESpecies> &pSpecies, const Deadline &pDue) {
        /*
         * If you made any guesses, you will find out the true species of those birds in this function.
         */

        //std::cerr << "REVEAL FUNCTION" << std::endl << std::endl;

        for (size_t i = 0; i < pSpecies.size(); i++) {

            //std::cerr << pSpecies[i] << std::endl;

            if (pSpecies[i] == -1) {
                movs_hist[6].push_back(pState.getBird((int) i));
            } else {
                movs_tolearn[pSpecies[i]].push_back(pState.getBird((int) i));
                size_t know = 0;

                for (size_t j = 0; j < pState.getBird(i).getSeqLength(); j++) {
                    if (pState.getBird(i).getObservation(j) == -1) {
                        break;
                    }
                    know++;
                }
                knowleadge[pSpecies[i]] += know;
                total_knowleadge++;
            }

        }

        //Refresh bird's models

        bb_models();

        std::cerr << std::endl << std::endl;
        std::cerr << "Accuracity = " << (double) dead_birds / shoots << std::endl;
        std::cerr << "Points for shooting = " << (int) (2 * dead_birds) - int(shoots) << std::endl;
        std::cerr << std::endl << std::endl;

    }

} /*namespace ducks*/
