//============================================================//
//                                                            //
//   This codes provides parallel omp codes to compute the    //
//  optimal fit for given molecular spectra, using the NLOPT  //
//     package. In addition it computes the optimal Raman     //
//  assisted excitation field to maximize discrimination in   //
//   a given ensemble of biological molecules for which the   //
//      excitation and Raman spectra are known or can be      //
//                  approximately obtained.                   //
//                                                            //
//                @author  A. Chattopadhyay                   //
//    @affiliation Princeton University, Dept. of Chemistry   //
//           @version Updated last on Dec 14 2018             //
//                                                            //
//============================================================//

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <nlopt.h>
#include <omp.h>
#include <time.h>
#define energy_factor 1. / 27.211385

typedef double complex cmplx;

typedef struct parameters{

    cmplx* rho_0;

    int nDIM;
    int nEXC;

    double* frequency_A;
    double* frequency_R;
    int freqDIM_A;
    int freqDIM_R;

    double* time_A;
    double* time_R;
    int timeDIM_A;
    int timeDIM_R;

    double field_amp_A;
    double field_amp_R;

    double omega_R;
    double omega_v;
    double omega_e;

    double exc_coeff_ratio;
    int thread_num;
    double* mu_guess;
    int mu_guess_num;
    double* freq_points;
    double* reference_spectra;
    double* Raman_levels;

} parameters;

typedef struct molecule{
    int nDIM;
    double* energies;
    double* gamma_population_decay;
    double* gamma_pure_dephasing;
    cmplx* rho_0;
    cmplx* mu;

    cmplx* field_A;
    cmplx* field_R;
    cmplx* rho;
    double* absorption_spectra;
    double* Raman_spectra;

    cmplx* dyn_rho_A;
    cmplx* dyn_rho_R;
} molecule;

typedef struct mol_system{
    molecule* mol;
    parameters* params;
} mol_system;


//====================================================================================================================//
//                                                                                                                    //
//                                        AUXILIARY FUNCTIONS FOR MATRIX OPERATIONS                                   //
//   ---------------------------------------------------------------------------------------------------------------  //
//    Given a matrix or vector and their dimensionality, these routines perform the operations of printing, adding,   //
//        scaling, copying to another compatible data structure, finding trace, or computing the maximum element.     //
//                                                                                                                    //
//====================================================================================================================//


void print_complex_mat(cmplx *A, int nDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX MATRIX                 //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e + %3.3eJ  ", creal(A[i * nDIM + j]), cimag(A[i * nDIM + j]));
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_complex_vec(cmplx *A, int vecDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX VECTOR                 //
//----------------------------------------------------//
{
	int i;
	for(i=0; i<vecDIM; i++)
	{
		printf("%3.3e + %3.3eJ  ", creal(A[i]), cimag(A[i]));
	}
	printf("\n");
}

void print_double_mat(double *A, int nDIM)
//----------------------------------------------------//
// 	            PRINTS A REAL MATRIX                  //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e  ", A[i * nDIM + j]);
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_double_vec(double *A, int vecDIM)
//----------------------------------------------------//
// 	          PRINTS A REAL VECTOR                    //
//----------------------------------------------------//
{
	int i;
	for(i=0; i<vecDIM; i++)
	{
		printf("%3.3e  ", A[i]);
	}
	printf("\n");
}


void copy_mat(const cmplx *A, cmplx *B, int nDIM)
//----------------------------------------------------//
// 	        COPIES MATRIX A ----> MATRIX B            //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] = A[i * nDIM + j];
        }
    }
}

void add_mat(const cmplx *A, cmplx *B, int nDIM)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] += A[i * nDIM + j];
        }
    }
}


void add_vec(const double *A, double *B, int nDIM)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        B[i] += A[i];
    }
}

void scale_mat(cmplx *A, double factor, int nDIM)
//----------------------------------------------------//
// 	     SCALES A BY factor ----> MATRIX B = A + B    //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        for(int j=0; j<nDIM; j++)
        {
            A[i * nDIM + j] *= factor;
        }
    }
}


void scale_vec(double *A, double factor, int nDIM)
//----------------------------------------------------//
// 	     SCALES A BY factor ----> VECTOR B = A + B    //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        A[i] *= factor;
    }
}


cmplx complex_trace(cmplx *A, int nDIM)
//----------------------------------------------------//
// 	                 RETURNS TRACE[A]                 //
//----------------------------------------------------//
{
    cmplx trace = 0.0 + I * 0.0;
    for(int i=0; i<nDIM; i++)
    {
        trace += A[i*nDIM + i];
    }
    printf("Trace = %3.3e + %3.3eJ  \n", creal(trace), cimag(trace));

    return trace;
}


double complex_abs(cmplx z)
//----------------------------------------------------//
// 	            RETURNS ABSOLUTE VALUE OF Z           //
//----------------------------------------------------//
{

    return sqrt((creal(z)*creal(z) + cimag(z)*cimag(z)));
}


double complex_max_element(cmplx *A, int nDIM)
//----------------------------------------------------//
// 	   RETURNS ELEMENT WITH MAX ABSOLUTE VALUE        //
//----------------------------------------------------//
{
    double max_el = A[0];
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            if(complex_abs(A[i * nDIM + j]) > max_el)
            {
                max_el = complex_abs(A[i * nDIM + j]);
            }
        }
    }
    return max_el;
}

double vec_max(const double *const A, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS ELEMENT WITH MAX ABSOLUTE VALUE        //
//----------------------------------------------------//
{
    double max_el = A[0];
    for(int i=0; i<nDIM; i++)
    {
        if(A[i] > max_el)
        {
            max_el = A[i];
        }
    }
    return max_el;
}

double vec_diff_norm(const double *const A, const double *const B, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS L-1 NORM OF VECTOR DIFFERENCE          //
//----------------------------------------------------//
{
    double norm=0.0;
    for(int i=0; i<nDIM; i++)
    {
        norm += fabs(A[i]-B[i]);
    }

    return norm;
}



//====================================================================================================================//
//                                                                                                                    //
//                   CALCULATION OF FIELDS FOR ABSORPTION AND RAMAN SPECTRA CALCULATIONS                              //
//                                                                                                                    //
//====================================================================================================================//

void CalculateAbsorptionSpectraField(molecule* mol, const parameters *const params, const int k)
//--------------------------------------------------------------------------------//
//     RETURNS THE ABSORPTION SPECTRA CALCULATION FIELD AS A FUNCTION OF TIME     //
//             k ----> index corresponding to spectral wavelength                 //
//--------------------------------------------------------------------------------//
{
    int i;
    int nDIM = params->nDIM;

    double* t = params->time_A;
    double A = params->field_amp_A;

    for(i=0; i<params->timeDIM_A; i++)
    {
        mol->field_A[i] = A * pow(cos(M_PI*t[i]/(fabs(2*t[0]))), 2) * cos(params->frequency_A[k] * t[i]);
    }
}

void CalculateRamanSpectraField(molecule* mol, const parameters *const params, const int k)
//--------------------------------------------------------------------------------//
//        RETURNS THE RAMAN SPECTRA CALCULATION FIELD AS A FUNCTION OF TIME       //
//             k ----> index corresponding to spectral wavelength                 //
//--------------------------------------------------------------------------------//
{
    int i;
    int nDIM = params->nDIM;
    int timeDIM_vib = params->timeDIM_R;

    double* t = params->time_R;

    double A_vib = params->field_amp_R;
    double w_R = params->omega_R;

    for(i=0; i<timeDIM_vib; i++)
    {
        mol->field_R[i] = A_vib * pow(cos(M_PI*t[i]/(fabs(2*t[0]))), 2) * (cos(w_R + params->frequency_R[k] * t[i]) + cos(w_R * t[i]));
    }
}


void CalculateRamanControlField(molecule* mol, const parameters *const params)
//---------------------------------------------------------------------------------//
//                 RETURNS THE RAMAN CONTROL FIELD AS A FUNCTION OF TIME           //
//---------------------------------------------------------------------------------//
{
    int i;
    int nDIM = params->nDIM;
    int timeDIM_vib = params->timeDIM_R;

    double* t = params->time_R;

    double A_vib = params->field_amp_R;
    double w_R = params->omega_R;

    for(i=0; i<timeDIM_vib; i++)
    {
        mol->field_R[i] = A_vib * pow(cos(M_PI*t[i]/(fabs(2*t[0]))), 2) * ((cos(w_R + params->omega_v * t[i])) + cos(w_R * t[i]));
    }
}


void CalculateExcitationControl(molecule* mol, const parameters *const params)
//---------------------------------------------------------------------------------//
//              RETURNS THE EXCITATION CONTROL FIELD AS A FUNCTION OF TIME         //
//---------------------------------------------------------------------------------//
{
    int i;
    int nDIM = params->nDIM;
    int timeDIM = params->timeDIM_A;

    double* t = params->time_A;

    double A = params->field_amp_A;

    for(i=0; i<timeDIM; i++)
    {
        mol->field_A[i] = A * pow(cos(M_PI*t[i]/(fabs(2*t[0]))), 2) * cos(params->omega_e * t[i]);
    }
}


//====================================================================================================================//
//                                                                                                                    //
//                                CALCULATION OF OPEN QUANTUM SYSTEM DYNAMICS                                         //
//                                                                                                                    //
//====================================================================================================================//


void L_operate(cmplx* Qmat, const cmplx field_ti, molecule* mol)
//----------------------------------------------------//
// 	    RETURNS Q <-- L[Q] AT A PARTICULAR TIME (t)   //
//----------------------------------------------------//
{
    int m, n, k;
    int nDIM = mol->nDIM;
    double* gamma_pure_dephasing = mol->gamma_pure_dephasing;
    double* gamma_population_decay = mol->gamma_population_decay;
    cmplx* mu = mol->mu;
    double* energies = mol->energies;

    cmplx* Lmat = (cmplx*)calloc(nDIM * nDIM,  sizeof(cmplx));

    for(m = 0; m < nDIM; m++)
        {
        for(n = 0; n < nDIM; n++)
            {
                Lmat[m * nDIM + n] += - I * (energies[m] - energies[n]) * Qmat[m * nDIM + n];
                for(k = 0; k < nDIM; k++)
                {
                    Lmat[m * nDIM + n] -= 0.5 * (gamma_population_decay[k * nDIM + n] + gamma_population_decay[k * nDIM + m]) * Qmat[m * nDIM + n];
                    Lmat[m * nDIM + n] += gamma_population_decay[m * nDIM + k] * Qmat[k * nDIM + k];
                    Lmat[m * nDIM + n] += I * field_ti * (mu[m * nDIM + k] * Qmat[k * nDIM + n] - Qmat[m * nDIM + k] * mu[k * nDIM + n]);
                }
                Lmat[m * nDIM + n] -= gamma_pure_dephasing[m * nDIM + n] * Qmat[m * nDIM + n];
            }

        }

    for(m = 0; m < nDIM; m++)
        {
        for(n = 0; n < nDIM; n++)
            {
                Qmat[m * nDIM + n] = Lmat[m * nDIM + n];
            }
        }
    free(Lmat);

}



//====================================================================================================================//
//                                                                                                                    //
//                                  PROPAGATION STEP FOR A GIVEN WAVELENGTH                                           //
//                                                                                                                    //
//====================================================================================================================//


void PropagateAbs(molecule* mol, const parameters *const params, const int indx)
//--------------------------------------------------------------------------------------------------------------------//
// 	 	 		       CALCULATES FULL LINDBLAD DYNAMICS  DUE TO THE CONTROL FIELD FROM TIME 0 to T               	  //
//                            indx gives index of the specific wavelength in the spectra                              //
//--------------------------------------------------------------------------------------------------------------------//
{

    int i, j, k;
    int tau_index, t_i;
    int nDIM = params->nDIM;
    int timeDIM_abs = params->timeDIM_A;

    cmplx *rho_0 = mol->rho_0;
    double *time = params->time_A;

    cmplx* field = mol->field_A;

    double dt = time[1] - time[0];

    cmplx* L_rho_func = (cmplx*)calloc(nDIM * nDIM, sizeof(cmplx));
    copy_mat(rho_0, L_rho_func, nDIM);
    copy_mat(rho_0, mol->rho, nDIM);

    for(t_i=0; t_i<timeDIM_abs; t_i++)
    {
        k=1;
        do
        {
            L_operate(L_rho_func, field[t_i], mol);
            scale_mat(L_rho_func, dt/k, nDIM);
            add_mat(L_rho_func, mol->rho, nDIM);
            k+=1;
        }while(complex_max_element(L_rho_func, nDIM) > 1.0E-6);

        copy_mat(mol->rho, L_rho_func, nDIM);
    }

    for(j=1; j<=params->nEXC; j++)
    {
        mol->absorption_spectra[indx] += mol->rho[(nDIM-j)*nDIM + (nDIM-j)];
    }
    free(L_rho_func);
}


void copy_molecule(molecule* original, molecule* copy, parameters* params)
//-----------------------------------------------------------------------------------//
//    MAKING AN ENSEMBLE OF DEEP COPIES OF AN INSTANCE OF THE MOLECULE STRUCTURE     //
//-----------------------------------------------------------------------------------//
{
    int N = params->mu_guess_num;
    int tid;
    int nDIM = params->nDIM;

    {
        for(int i=0; i<N; i++)
        {
            copy[i].energies = (double*)malloc(N*sizeof(double));
            copy[i].nDIM = nDIM;
            copy[i].gamma_population_decay = (double*)malloc(nDIM*nDIM*sizeof(double));
            copy[i].gamma_pure_dephasing = (double*)malloc(nDIM*nDIM*sizeof(double));
            copy[i].rho_0 = (cmplx*)malloc(nDIM*nDIM*sizeof(cmplx));
            copy[i].mu = (cmplx*)malloc(nDIM*nDIM*sizeof(cmplx));
            copy[i].field_A = (cmplx*)malloc(params->timeDIM_A*sizeof(cmplx));
            copy[i].field_R = (cmplx*)malloc(params->timeDIM_R*sizeof(cmplx));
            copy[i].rho = (cmplx*)malloc(nDIM*nDIM*sizeof(cmplx));
            copy[i].absorption_spectra = (double*)malloc(params->freqDIM_A*sizeof(double));
            copy[i].Raman_spectra = (double*)malloc(params->freqDIM_R*sizeof(double));
            copy[i].dyn_rho_A = (cmplx*)malloc(nDIM*params->timeDIM_A*sizeof(cmplx));
            copy[i].dyn_rho_R = (cmplx*)malloc(nDIM*params->timeDIM_R*sizeof(cmplx));

            for(int k=0; k<params->nEXC; k++)
            {
                copy[i].energies[k+(params->nDIM - params->nEXC)] = params->freq_points[4*i+k]*energy_factor;
            }

            for(int k=0; k<params->nDIM - params->nEXC; k++)
            {
                copy[i].energies[k] = params->Raman_levels[k];
            }

            memcpy(copy[i].gamma_population_decay, original->gamma_population_decay, nDIM*nDIM*sizeof(double));
            memcpy(copy[i].gamma_pure_dephasing, original->gamma_pure_dephasing, nDIM*nDIM*sizeof(double));
            memcpy(copy[i].rho_0, original->rho_0, nDIM*nDIM*sizeof(cmplx));
            memcpy(copy[i].mu, original->mu, nDIM*nDIM*sizeof(cmplx));
            scale_mat(copy[i].mu, params->mu_guess[i], nDIM);
            memcpy(copy[i].field_A, original->field_A, params->timeDIM_A*sizeof(cmplx));
            memcpy(copy[i].field_R, original->field_R, params->timeDIM_A*sizeof(cmplx));
            memcpy(copy[i].rho, original->rho, nDIM*nDIM*sizeof(cmplx));
            memcpy(copy[i].absorption_spectra, original->absorption_spectra, params->freqDIM_A*sizeof(double));
            memcpy(copy[i].Raman_spectra, original->Raman_spectra, params->freqDIM_R*sizeof(double));
            memcpy(copy[i].dyn_rho_A, original->dyn_rho_A, nDIM*params->timeDIM_A*sizeof(cmplx));
            memcpy(copy[i].dyn_rho_R, original->dyn_rho_R, nDIM*params->timeDIM_R*sizeof(cmplx));

        }
    }
}


cmplx* CalculateSpectra(molecule* molA, molecule* molB, parameters* params)
//--------------------------------------------------------------------//
//        GETTING rho(T) FROM rho(0) USING PROPAGATE FUNCTION         //
//--------------------------------------------------------------------//
{
    int tid;

    scale_mat(molB->mu, params->exc_coeff_ratio, params->nDIM);
    copy_mat(molA->rho_0, molA->rho, params->nDIM);
    copy_mat(molB->rho_0, molA->rho, params->nDIM);

    omp_set_num_threads(params->thread_num);
    molecule* ensemble = malloc(params->mu_guess_num * sizeof(molecule));

    copy_molecule(molA, ensemble, params);
    print_double_vec(ensemble[2].energies, params->nDIM);

    #pragma omp parallel for
    for(int j=0; j<params->mu_guess_num; j++)
        for(int i=0; i<params->freqDIM_A; i++)
        {
            memset(ensemble[j].absorption_spectra, 0, params->freqDIM_A*sizeof(double));
            CalculateAbsorptionSpectraField(&ensemble[j], params, i);
            PropagateAbs(&ensemble[j], params, i);
            add_vec(ensemble[j].absorption_spectra, molA->absorption_spectra, params->freqDIM_A);
        }
    free(ensemble);
    scale_vec(molA->absorption_spectra, 100./vec_max(molA->absorption_spectra, params->freqDIM_A), params->freqDIM_A);
    printf("The norm of fit is %g \n", vec_diff_norm(params->reference_spectra, molA->absorption_spectra, params->freqDIM_A));
}



































//double nlopt_spectra_func(unsigned N, const double *spectra_params, double *grad_J, void *nloptJ_params)
//{
//    mol_system** Ensemble = (mol_system**)nloptJ_params;
//
//    parameters* params = (*Ensemble)->params;
//    molecule* mol = (*Ensemble)->mol;
//
//    double J;
//
//    for(int i=0; i<=6; i++)
//    {
//        params->mu_guess[i] = spectra_params[i];
//    }
//
//    int i, chunk, tid;
//
//    chunk = CHUNKSIZE;
//
//    omp_set_num_threads(omp_get_num_procs());
//
//    #pragma omp parallel shared(a, b, chunk) private(i)
//    {
//    #pragma omp for schedule(dynamic, chunk)
//        for (i=0; i< N; i++)
//          {
//            b[i] = myfun(a[i]);
//          }
//    }
//
//    int nDIM = params->nDIM;
//
//    return J;
//}